""" 
ORBIT dataloader for the benchmark formulation:

- This script borrows certain functions from (Massiceti et al. (2021)) to accompany the benchmark 

"""

# Libraries
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
import torchvision.transforms as transforms
import json 
import random 
import torch 
import os 

# 
from .orbit_dataset import UserEpisodicORBITDataset
import numpy as np 

# Path for Orbit Dataset
ORBIT_DATASET_PATH = '/fs/cml-datasets/ORBIT/orbit_benchmark_224'

# Initialise the dataset for ORBIT
def init_dataset(args):
    # Preload clips
    preload = not args.no_preload_clips
    # Dataset info
    dataset_info = {
            'mode': 'test', # mode is always test
            'data_path': args.data_path,
            'train_object_cap': args.train_object_cap,
            'with_train_shot_caps': args.with_train_shot_caps,
            'with_cluster_labels': False,
            'train_way_method' : args.train_way_method,
            'test_way_method' : args.test_way_method,
            'train_shot_methods' : [args.train_context_shot_method, args.train_target_shot_method],
            'test_shot_methods' : [args.test_context_shot_method, args.test_target_shot_method],
            'train_tasks_per_user': args.train_tasks_per_user,
            'test_tasks_per_user': args.test_tasks_per_user,
            'train_task_type' : args.train_task_type,
            'test_set': 'test',
            'shots' : [args.context_shot, args.target_shot],
            'video_types' : [args.context_video_type, args.target_video_type],
            'clip_length': args.clip_length,
            'train_num_clips': [args.train_context_num_clips, args.train_target_num_clips],
            'test_num_clips': [args.test_context_num_clips, args.test_target_num_clips],
            'subsample_factor': args.subsample_factor,
            'frame_size': args.frame_size,
            'annotations_to_load': args.annotations_to_load,
            'preload_clips': preload,
        }


    # Information about the dataset


    return dataset_info



""" 
ORBIT dataset class
- has sampler which samples in the following sequence: Sample user ====> sample objects (Number of ways) =====> sample videos (Number of Shots) ====> For every video, choose an appropriate number of frames
Original sampling: For every video ====> Sample a certain number of clips and for every clip obtain 8 frames which are averaged out in the feature extractor
Our sampling process for creating the search support pool and query:

############ Option 1: Get hard episodes per user ####################
===> Step 1: Sample a user
===> Step 2: Get the entire set of objects for the user ==> Ways / Number of classes (Sample the number of ways)
===> Step 3: Each class, will have multiple videos ==> sample the query set(vid1, vid2)[Randomly sample] ===> Sample the support pool(vid3, vid4) 
    ===> Join all the frames which have an object in it and that becomes the search pool 

############ Option 2: Get hard episodes agnostic to user ############


"""

class DatasetQueue:
    """
    Class for a queue of tasks sampled from UserEpisodicORIBTDataset/ObjectEpisodicORBITDataset.
    """
    def __init__(self, tasks_per_user, shuffle, test_mode, override_num_workers = None):
        """
        Creates instance of DatasetQueue.
        :param tasks_per_user: (int) Number of tasks per user to add to the queue.
        :param shuffle: (bool) If True, shuffle tasks, else do not shuffled.
        :param test_mode: (bool) If True, only return target set for first task per user.
        :param num_workers: (Optional[int]) Number of workers to use. Overrides defaults (4 if test, 8 otherwise).
        :return: Nothing.
        """

        # Number of tasks per user
        self.tasks_per_user = tasks_per_user
        self.shuffle = shuffle
        self.test_mode = test_mode
        if override_num_workers is None:
            self.num_workers = 4 if self.test_mode else 8
        else:
            self.num_workers = override_num_workers

        self.num_users = None
        self.collate_fn = self.unpack

    def unpack(self, batch):
        #assumes batch_size = 1
        assert len(batch) == 1, "DataLoader needs a batch size of 1!"
        unpacked_batch = {}
        for k,v in batch[0].items():
            unpacked_batch[k] = v
        return unpacked_batch

    # Number of users
    def get_num_users(self):
        return self.num_users

    def get_cluster_classes(self):
        return self.dataset.cluster_classes

    # Primary data-loaders
    def get_tasks(self):
        return torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=False,
                num_workers=self.num_workers,
                sampler=TaskSampler(self.tasks_per_user, self.num_users, self.shuffle, self.test_mode),
                collate_fn=self.collate_fn
                )

# 
class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        # Dataset queue
        DatasetQueue.__init__(self, tasks_per_user, shuffle, test_mode)
        self.dataset = UserEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users



""" 
Class which samples the query frames and the pool of support frames to search from
- the sampling is done per user (domain) when compared to Meta-Dataset
"""

class ORBITDataset(VisionDataset):
    def __init__(self, root, args, transform=None, target_transform=None, transforms=None, img_format="jpg"):
        """Init ObjectNet pytorch dataloader."""
        super(ORBITDataset, self).__init__(root, args)    
        print("Path for the orbit dataset: {}".format(root))

        # Get the dataset info
        dataset_info = init_dataset(args)

        # Dataset Info
        self.data_info = dataset_info 

        # Test queue Class which inherits
        self.test_queue = self.config_user_centric_queue(
                                        os.path.join(dataset_info['data_path'], dataset_info['test_set']),
                                        dataset_info['test_way_method'],
                                        'max', #object cap during testing
                                        dataset_info['test_shot_methods'],
                                        dataset_info['shots'],
                                        dataset_info['video_types'],
                                        dataset_info['subsample_factor'],
                                        dataset_info['test_num_clips'],
                                        dataset_info['clip_length'],
                                        dataset_info['preload_clips'],
                                        dataset_info['frame_size'],
                                        dataset_info['annotations_to_load'],
                                        dataset_info['test_tasks_per_user'],
                                        test_mode=True)
        print("#############################################################")
        print("#############################################################")

        # Dataset Information
        print("Dataset Information: {}".format(self.data_info))

        # Cap on the number of classes 
        self.way_opt = 5

        # Cap on the number of frames / class for the support pool 
        self.support_cap_per_video = 300

        # Cap on the number of frames / class for the target
        self.target_cap_per_video = 20

        
        return 
    
    # choose the videos
    def choose_videos(self, videos, test_dataset, type = 'context'):
        if type == 'context':
            # initialise params
            required_shots = test_dataset.shot_context
            shot_method = test_dataset.shot_method_context 
            shot_cap = test_dataset.context_shot_cap
        else:
            required_shots = test_dataset.shot_target 
            shot_method = test_dataset.shot_method_target 
            shot_cap = test_dataset.target_shot_cap 
        
        # Cap for memory purpose
        required_shots = min(required_shots, shot_cap)
        num_videos = len(videos)
        available_shots = min(required_shots, num_videos)

        # 
        if shot_method == 'specific': # sample specific videos (1 required_shots = 1st video; 2 required_shots = 1st, 2nd videos, ...)
            return videos[:available_shots]
        elif shot_method == 'fixed': # randomly sample fixed number of videos
            return random.sample(videos, available_shots)
        elif shot_method == 'random': # randomly sample a random number of videos between 1 and num_videos
            max_shots = min(num_videos, shot_cap) # capped for memory reasons
            random_shots = random.choice(range(1, max_shots+1))
            return random.sample(videos, random_shots)
        
        ########### This is chosen as all the videos will be used until the shot cap #############
        elif shot_method == 'max': # samples all videos
            max_shots = min(num_videos, shot_cap) # capped for memory reasons
            return random.sample(videos, max_shots)
    


    """
    Function for splitting the videos into context and target
    - test_dataset: Dataset class for ORBIT 
    - videos: Path for the videos of the object {"clean": [], "clutter": []}
    - context_type / target_type : Clean / Clutter to sample from 
    - Return: (list:: str, list::str) Sampled context and target video paths for given object
    """
    def sample_videos(self, test_dataset, videos, context_type='clean', target_type = 'clean'):
        # Context type: Clean and Target type: Clean
        if context_type == 'clean' and target_type == 'clean':
            # Number of available clean videos
            num_context_avail = len(videos['clean'])
            split = min(5, num_context_avail-1) # Maximum of 5 videos for context and the rest for query; unless left
            #print(f'Total videos: {num_context_avail} out of which {split} are selected.')
            
            # Choose the appropriate videos for context and query
            context = self.choose_videos(videos['clean'][:split], test_dataset, 'context')
            target = self.choose_videos(videos['clean'][split:], test_dataset, 'target')

        else:
            # Context type: Clean + Clutter Set, Target type: Clean
            context = self.choose_videos(videos['clean'][:split] + videos['clutter'][:split], test_dataset, 'context')
            target = self.choose_videos(videos['clean'][split:], test_dataset, 'target')
        

        return context, target 
    
    
    """ 
    - task_objects: Objects selected for the user
    - with_target_set: Index of the user
    - user_id: Current user 
    - video2id:  video_path ===> video_id
    - vid2frames: video_path ===> [Path of all the frames in the video]
    - obj2vids: [List]: Each index is objectId and the corresponding entry is a dictionary {"clean": [video_paths], "clutter": [video_paths]}
    - obj2name: [List] where each index is the object_id
    - test_dataset: ORBIT dataset class

    """
    def sample_task(self, task_objects, with_target_set, user_id, video2id, vid2frames, obj2vids, obj2name, test_dataset):
        # Step 1: Select ways
        # Select way
        num_objects = len(task_objects)
        way = test_dataset.compute_way(num_objects)
        selected_objects = sorted(random.sample(task_objects, way))
        label_map = test_dataset.get_label_map(selected_objects, test_dataset.with_cluster_labels)


        # Set caps, for memory purposes
        if test_dataset.with_caps:
            self.context_shot_cap = 5 if way>=6 else 10 
            self.target_shot_cap = 4 if way>=6 else 8
        
        # Object List
        obj_list = []
        context_clips, target_clips = [], []
        context_paths, target_paths = [], []
        context_labels, target_labels = [], []
        context_annotations, target_annotations = [], []


        # TODO: Fill it up

        # Annotations ===> Context
        # context_annotations = {ann: [] for ann in test_dataset.annotations_to_load}
        # # Annotations ===> Target
        # target_annotations = {ann: [] for ann in test_dataset.annotations_to_load}
        
        # Context / Target number of clips
        context_num_clips = test_dataset.context_num_clips # By default random 
        target_num_clips = test_dataset.target_num_clips # By default max


        print("Number of context: {}".format(context_num_clips))
        print("Target number of clips: {}".format(target_num_clips))

        # List of objects
        obj_list = []
        obj_count = 0

        # Make it a 5-way task
        selected_objects = selected_objects[:self.way_opt]

        # Go through each of the object 
        for obj in selected_objects:
            label = label_map[obj]
            obj_name = obj2name[obj]
            obj_list.append(obj_name)
            
            ######### Selecting the support and query #########
            # Context videos, target videos --- Non-overlapping sets of videos in the context and target
            context_videos, target_videos = self.sample_videos(test_dataset, obj2vids[obj])

            # Sample context and target clips from the videos 
            context_clips_, context_paths_, annotations_context = self.sample_clips_from_videos(test_dataset, context_videos, context_num_clips, video_type = 'context')
            target_clips_, target_paths_, annotations_target = self.sample_clips_from_videos(test_dataset, target_videos, target_num_clips, video_type = 'target')

            context_labels.extend([label for _ in range(len(context_clips_))])
            target_labels.extend([label for _ in range(len(target_clips_))]) 
            # Append to context_clips
            context_clips.extend(context_clips_)
            target_clips.extend(target_clips_)
            
            # Path of the contexts / target
            context_paths.extend(context_paths_)
            target_paths.extend(target_paths_)

            context_annotations.extend(annotations_context)
            target_annotations.extend(annotations_target)

        
        

        ## Stack onto torch tensor ##
        context_clips = torch.stack(context_clips)
        target_clips = torch.stack(target_clips)
        context_labels = torch.tensor(context_labels).reshape(1,-1)
        target_labels = torch.tensor(target_labels).reshape(1,-1)

        
        
        return context_clips, context_labels, target_clips, target_labels, context_paths, target_paths, context_annotations, target_annotations


    """
    Samples clips from a video-path
    - test_dataset: Original ORBIT dataset class
    - video_path: Path of the video 
    - num_clips: Number of clips to sample from: If max ==> Sample all non-overlapping clips. If random ==> sample random number of non-overlapping clips.

    Returns: A subset of the paths for the frames inside the video (Frame paths organised in clips of self.clip_length contiguous frames)
    """
    def sample_clips_from_a_video(self, test_dataset, video_path, num_clips):        
        # Total frame paths for all the videos. (All the frame paths for the video_path)
        frame_paths = test_dataset.vid2frames[video_path]
        
        # Subsampled frame paths
        subsampled_frame_paths = frame_paths[0:test_dataset.frame_cap: test_dataset.subsample_factor] # TODO
        spare_frames = len(subsampled_frame_paths) % test_dataset.clip_length
        subsampled_frame_paths.extend([subsampled_frame_paths[-1]] * (test_dataset.clip_length - spare_frames))

        num_subsampled_frames = len(subsampled_frame_paths)
        max_num_clips = num_subsampled_frames // test_dataset.clip_length 
        
        assert num_subsampled_frames % test_dataset.clip_length == 0

        # For query
        if num_clips == 'max':
            #sampled_paths = subsampled_frame_paths[:max_num_clips*test_dataset.clip_length]
            ####### Paths for the query ====> Filtering step required #######
            #sampled_paths = np.array(sampled_paths).reshape((max_num_clips, test_dataset.clip_length))
            sampled_paths = np.array(subsampled_frame_paths)

            # TODO:
            #ubsample_sampled_paths()

        # Sampling for context
        else:
            # Sampled paths
            sampled_paths = np.array(subsampled_frame_paths)

            # TODO : Filtering for the paths with only objects

    

        return sampled_paths
    
    """ 
    Function to load clips from the disk paths to tensors
    Frame paths are organized in the general form at this moment : TODO change this depending on the frame selected per clip
    """
    # Function to load the clips 
    def load_clips(self, paths, test_dataset):
        # Loaded clips
        #loaded_clips = torch.zeros(len(paths), 3, test_dataset.frame_size, test_dataset.frame_size)
        loaded_clips = torch.zeros(len(paths), 3, 128, 128)

        for clip_idx in range(0, len(paths)):
            frame_path = paths[clip_idx]
            loaded_clips[clip_idx] = test_dataset.load_and_transform_frame(frame_path)


        return loaded_clips

    """
    Function to load the appropriate annotatiosn from the sampled paths
    - test_dataset: ORBIT dataset class
    - sampled_paths: 
    """
    def load_annotations(self, test_dataset, sampled_paths):
        # Storage of the annotations 
        annotations_storage = []

        # For every path sample the annotations
        for path in sampled_paths:
            # Get the frame path
            frame_name = os.path.basename(path)
            # Get the frame name as there is a mapping from frame_name ===> annotations
            frame_anns = test_dataset.frame2anns[frame_name]
            annotations_storage.append(frame_anns)
        

        return annotations_storage
    
    
    """
    Filtering frames based on no_object present issue --- To have frames which have objects
    - sampled_paths: Paths selected for the video
    - sampled_annotations: Annotations for the video
    """
    def filter(self, sampled_paths, sampled_annotations):
        # Filtered paths
        filtered_sampled_paths = []

        # Total number of issues
        issues = ['object_not_present_issue', 'framing_issue', 'viewpoint_issue', 'blur_issue', 'occlusion_issue', 'overexposed_issue', 'underexposed_issue']
        issue_dict = {v: 0 for v in issues}
        annotation_total = []


        # Mine the frames which have an object
        for i in range(0, len(sampled_paths)):
            # Current annotation
            curr_annotation = sampled_annotations[i]

            # If the object is present ===> Current annotation
            if curr_annotation['object_not_present_issue'] == False:
                filtered_sampled_paths.append(sampled_paths[i])
                annotation_total.append(curr_annotation)
            
            for issue in issues:
                if curr_annotation[issue] == True:
                    issue_dict[issue] += 1

        
        # Filtered sampled paths
        filtered_sampled_paths = np.array(filtered_sampled_paths)

        # Return the filtered sampled_paths, issue_dict
        return filtered_sampled_paths, issue_dict, annotation_total 
    

    # Filtering the target with annotations
    def filter_target(self, sampled_paths, sampled_annotations):
        # Resampled paths
        resampled_paths = []
        resampled_annotations = []
        count = 0
        for i in range(0, len(sampled_annotations)):
            if sampled_annotations[i]['viewpoint_issue'] == False: #and sampled_annotations[i]['occlusion_issue'] == False:
                resampled_paths.append(sampled_paths[i])
                resampled_annotations.append(sampled_annotations[i])

                count += 1
        
        # If count is zero (i.e. )
        if count == 0:
            for i in range(0, len(sampled_annotations)):
                resampled_paths.append(sampled_paths[i])
                resampled_annotations.append(sampled_annotations[i])


        # Resampled Paths
        return resampled_paths, resampled_annotations


    """ 
    Function to sample clips from videos:
    - video_paths: Paths of all the videos for the set from which the clips need to be extracted
    - num_clips: Clip sampling strategy (Original : Random for support, Max for query)
    """
    def sample_clips_from_videos(self, test_dataset, video_paths, num_clips, video_type = 'context'):
        # For context type --- Get all frames --> Filter ---> Subsample ---> Join together to form the support pool
        clips, paths, video_ids = [], [], []

        # Stores the total annotations
        annotations_total = []

        # TODO
        # Annotation dictionary
        annotations = {ann: [] for ann in test_dataset.annotations_to_load}
        
        # For each of the video paths -- sample clips / frames
        for video_path in video_paths:
            if video_type == 'context':
                # Sampling path of the frames from the video_path ===> Context (Support): num_clips will be chosen appropriately
                sampled_paths = self.sample_clips_from_a_video(test_dataset, video_path, num_clips=None)
            
            else:
                # Sampling path of the frames from the video_path ===> Target (Query): num_clips ==> max
                sampled_paths = self.sample_clips_from_a_video(test_dataset, video_path, num_clips)
            

            # Check if annotations need to be loaded
            if test_dataset.with_annotations:
                # Randomize the sampled paths - per video
                random.shuffle(sampled_paths)
                if video_type == 'context':
                    # Sampled Paths for the Context Set
                    sampled_paths = sampled_paths[:self.support_cap_per_video] # Sample support_cap_per_video frames / class
                
                else:
                    # Sampled Paths for the Target Set
                    sampled_paths = sampled_paths[:self.target_cap_per_video]

                # Sampled annotations
                sampled_annotations = self.load_annotations(test_dataset, sampled_paths)
                
                # Add the step of using only the sampled paths of query sets without issues
                if video_type == 'target':
                    sampled_paths, sampled_annotations = self.filter_target(sampled_paths, sampled_annotations)
                    sampled_paths = sampled_paths[:self.target_cap_per_video]
                    sampled_annotations = sampled_annotations[:self.target_cap_per_video]


                # Filtered sampled paths ---- Only using frames where the object is present
                sampled_paths, issue_dict, annotation_sampled = self.filter(sampled_paths, sampled_annotations)


            # Extend the sampled paths --- paths of the frames which are chosen after integration
            paths.extend(sampled_paths)
            annotations_total.extend(annotation_sampled)


            # Pre-loading of the clips
            if test_dataset.preload_clips:
                sampled_clips = self.load_clips(sampled_paths, test_dataset)
                clips += sampled_clips
                
            
            # TODO : Annotations integration
            

        return clips, paths, annotations_total
    


    """ 
    Sampler for ORBIT dataset which returns:
    - support_frames: For every video ==>  Join all the frames from the given videos ===> Filter on no_object present ===> Subsample Frames at particular intervals ====> Form the support pool 
    - support_labels: Label for the object
    - query_images: Filter on no_object present ===> Join the frames to get the query pool
    - query_labels: Labels for the object
    - support_paths: Support paths for the selected frames 
    - query_paths : Query paths for the selected frames
    - annotations of each frame: Annotations for each of the frame
    """
    def sampler(self, args):
        # Class for the dataset
        test_dataset = self.test_queue.dataset

        # Users from which the sampling needs to be done
        self.users = test_dataset.users 
        curr_user = random.sample(self.users, 1)[0] # Sample one user 
        print("Current user: {}".format(curr_user))
        # User objects (Ids of the objects)
        user_objects = test_dataset.user2objs[curr_user]
        object_list = [test_dataset.obj2name[obj_id] for obj_id in user_objects]
        num_objects = len(object_list)
        # Number of objects
        print("Number of objects: {}".format(num_objects))

        ######### Important initialisations #######
        video2id = test_dataset.video2id # video_path ===> video_id
        vid2frames = test_dataset.vid2frames # video_path ===> [Path of all the frames in the video]
        obj2vids = test_dataset.obj2vids # [List]: Each index is objectId and the corresponding entry is a dictionary {"clean": [video_paths], "clutter": [video_paths]}
        obj2name = test_dataset.obj2name # [List] where each index is the object_id
        
        #### Step 1: Sample the ways ####
        task_objects = user_objects
        user_id = curr_user
        with_target_set = self.users.index(user_id)

        # Context Clips / Labels / Paths; Target Clips / Labels / Paths
        context_clips, context_labels, target_clips, target_labels, context_paths, target_paths, context_annotations, target_annotations = self.sample_task(task_objects, with_target_set, user_id, video2id, vid2frames, obj2vids, obj2name, test_dataset)

        context_clips = torch.unsqueeze(context_clips, dim = 0)
        target_clips = torch.unsqueeze(target_clips, dim = 0)

        ###################################### Testing for no_object_issue_present #####################################
        # users = test_dataset.users
        # for user in users:
        #     count = 0
        #     user_objects = test_dataset.user2objs[user]
            
        #     for obj_id in user_objects:
        #         videos = test_dataset.obj2vids[obj_id]['clean']
        #         for vid in videos:
        #             frames = test_dataset.vid2frames[vid]
        #             annotations = self.load_annotations(test_dataset, frames)
                    
        #             for annotation in annotations:
        #                 if annotation['object_not_present_issue'] == True:
        #                     count += 1
    

        #     print(f'For user: {user}, there are {count} frames where issue is present')
        ################################################################################################################

        return context_clips, context_labels, target_clips, target_labels, context_paths, target_paths, context_annotations, target_annotations
    
    # User-centric queue
    def config_user_centric_queue(self, root, way_method, object_cap, shot_method, shots, video_types, \
                            subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, \
                            tasks_per_user, test_mode=False, with_cluster_labels=False, with_caps=False, shuffle=False):
        
        # Tasks per user ==> 5
        # annotations_to_load ==> Check this argument
        # clip length 
        


        # Return argument
        return UserEpisodicDatasetQueue(root, way_method, object_cap, shot_method, shots, video_types, \
                                subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, \
                                tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle)
