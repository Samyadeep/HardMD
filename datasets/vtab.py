""" 
Loader for VTAB dataset for two tasks:

(a) Sampler Pipeline for Support Set Extraction

(b) Data-loader for few-shot fine-tuning

"""


# # Libraries
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
import torchvision.transforms as transforms
import json 
import random 
import torch 
import torchvision.transforms as T
import tensorflow as tf
import tensorflow_datasets as tfds
#import tfds_nightly as tfds
import numpy as np

from tensorflow_datasets.core.utils import gcs_utils

# gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
# gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

# Tensorflow dataset reader
class TfDatasetReader:
    def __init__(self, dataset, task, context_batch_size, target_batch_size, path_to_datasets, image_size, device):
        self.dataset = dataset
        self.task = task
        self.device = device
        self.image_size = image_size
        self.context_batch_size = context_batch_size
        self.target_batch_size = target_batch_size
        tf.compat.v1.enable_eager_execution()

        train_split = 'train[:{}]'.format(context_batch_size)
        ds_context, ds_context_info = tfds.load(
            dataset,
            split=train_split,
            shuffle_files=True,
            data_dir=path_to_datasets,
            with_info=True,
            try_gcs=True
        )
        self.context_dataset_length = ds_context_info.splits["train"].num_examples
        self.context_iterator = ds_context.as_numpy_iterator()

        test_split = 'test'
        if self.dataset == 'clevr':
            test_split = 'validation'
        if 'test' in ds_context_info.splits:
            # we use the entire test set
            ds_target, ds_target_info = tfds.load(
                dataset,
                split=test_split,
                shuffle_files=False,
                data_dir=path_to_datasets,
                with_info=True,
                try_gcs = True
                )
            self.target_dataset_length = ds_target_info.splits["test"].num_examples
        else:  # there is no test split
            # get a second iterator to the training set and skip the training examples
            test_split = 'train[{}:]'.format(context_batch_size)
            ds_target = tfds.load(
                dataset, split=test_split,
                shuffle_files=False,
                data_dir=path_to_datasets,
                try_gcs=True
            )
            self.target_dataset_length = self.context_dataset_length - context_batch_size
        self.target_iterator = ds_target.as_numpy_iterator()

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        ])

    def get_context_batch(self):
        return self._get_batch(self.context_iterator, is_target=False)

    def get_target_batch(self):
        return self._get_batch(self.target_iterator, is_target=True)

    def get_context_dataset_length(self):
        return self.context_dataset_length

    def get_target_dataset_length(self):
        return self.target_dataset_length

    def _get_batch(self, iterator, is_target):
        batch_size = self.target_batch_size if is_target else self.context_batch_size
        images = []
        labels = []
        for i in range(batch_size):
            try:
                item = iterator.next()
            except StopIteration:  # the last batch may be less than batch_size
                break

            # images
            images.append(self._prepare_image(item['image']))

            # labels
            if self.dataset == "clevr":
                labels.append(self._get_clevr_label(item, self.task))
            elif self.dataset == 'kitti':
                labels.append(self._get_kitti_label(item))
            elif self.dataset == 'smallnorb':
                if self.task == 'azimuth':
                    labels.append(item['label_azimuth'])
                elif self.task == 'elevation':
                    labels.append(item['label_elevation'])
                else:
                    raise ValueError("Unsupported smallnorb task.")
            elif self.dataset == "dsprites":
                labels.append(self._get_dsprites_label(item, self.task))
            else:
                labels.append(item['label'])

        labels = np.array(labels)
        images = torch.stack(images)

        # move the images and labels to the device
        images = images.to(self.device)
        labels = torch.from_numpy(labels)
        if is_target:
            labels = labels.type(torch.LongTensor).to(self.device)
        else:
            labels = labels.to(self.device)

        return images, labels

    def _get_kitti_label(self, x):
        """Predict the distance to the closest vehicle."""
        # Location feature contains (x, y, z) in meters w.r.t. the camera.
        vehicles = np.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
        vehicle_z = np.take(x["objects"]["location"][:, 2], vehicles)
        if len(vehicle_z.shape) > 1:
            vehicle_z = np.squeeze(vehicle_z, axis=0)
        if vehicle_z.size == 0:
            vehicle_z = np.array([1000.0])
        else:
            vehicle_z = np.append(vehicle_z, [1000.0], axis=0)
        dist = np.amin(vehicle_z)
        # Results in a uniform distribution over three distances, plus one class for "no vehicle".
        thrs = np.array([-100.0, 8.0, 20.0, 999.0])
        label = np.amax(np.where((thrs - dist) < 0))
        return label

    def _get_dsprites_label(self, item, task):
        num_classes = 16
        if task == "location":
            predicted_attribute = 'label_x_position'
            num_original_classes = 32
        elif task == "orientation":
            predicted_attribute = 'label_orientation'
            num_original_classes = 40
        else:
            raise ValueError("Bad dsprites task.")

        # at the desired number of classes. This is useful for example for grouping
        # together different spatial positions.
        class_division_factor = float(num_original_classes) / float(num_classes)

        return np.floor((item[predicted_attribute]) / class_division_factor).astype(int)

    def _get_clevr_label(self, item, task):
        if task == "count":
            label = len(item["objects"]["size"]) - 3
        elif task == "distance":
            dist = np.amin(item["objects"]["pixel_coords"][:, 2])
            # These thresholds are uniformly spaced and result in more or less balanced
            # distribution of classes, see the resulting histogram:
            thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
            label = np.amax(np.where((thrs - dist) < 0))
        else:
            raise ValueError("Bad clevr task.")

        return label

    def _prepare_image(self, image):
        if self.dataset == "smallnorb" or self.dataset == "dsprites":
            # grayscale images where the channel needs to be squeezed to keep PIL happy
            image = np.squeeze(image)

        if self.dataset == "dsprites":  # scale images to be in 0 - 255 range to keep PIL happy
            image = image * 255.0

        im = Image.fromarray(image)
        im = im.resize((self.image_size, self.image_size), Image.LANCZOS)
        im = im.convert("RGB")
        return self.transforms(im)



# VTAB dataset class
class VTAB(VisionDataset):
    """
    ObjectNet dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
    """

    def __init__(self, root, context_set_size = 1000, test_batch_size = 40):
        """Init ObjectNet pytorch dataloader."""
        super(VTAB, self).__init__(root)

        # Size of the context
        self.context_set_size = context_set_size

        # 18 Datasets for VTABv2 (with DTD removed)
        self.datasets = [
            # {'name': "caltech101", 'task': None, 'enabled': True}, # 3060 examples 
            {'name': "cifar100", 'task': None, 'enabled': True}, # 
            {'name': "oxford_flowers102", 'task': None, 'enabled': True},
            {'name': "oxford_iiit_pet", 'task': None, 'enabled': True},
            {'name': "sun397", 'task': None, 'enabled': True},
            {'name': "svhn_cropped", 'task': None, 'enabled': True},
            {'name': "eurosat", 'task': None, 'enabled': True},
            {'name': "resisc45", 'task': None, 'enabled': True},
            {'name': "patch_camelyon", 'task': None, 'enabled': True},
            {'name': "diabetic_retinopathy_detection", 'task': None, 'enabled': True},
            {'name': "clevr", 'task': "count", 'enabled': True},
            {'name': "clevr", 'task': "distance", 'enabled': True},
            {'name': "dsprites", 'task': "location", 'enabled': True},
            {'name': "dsprites", 'task': "orientation", 'enabled': True},
            {'name': "smallnorb", 'task': "azimuth", 'enabled': True},
            {'name': "smallnorb", 'task': "elevation", 'enabled': True},
            {'name': "dmlab", 'task': None, 'enabled': True},
            {'name': "kitti", 'task': None, 'enabled': True},
        ]

        # Natural tasks
        natural_tasks = ['caltech101', 'cifar100', 'oxford_flowers102', 'oxford_iiit_pet', 'svhn_cropped'] # 5-datasets
        specialized_tasks = ['patch_camelyon', 'eurosat', 'resisc45'] # 3-datasets
        structured_tasks = ['clevr', 'dmlab', 'dsprites', 'kitti', 'smallnorb'] # 2 + 1 + 2 + 1 + 2 = 8-datasets

        # 2 datasets are diabetic_retinopathy and sun397 which need to be installed

        # Size of context set
        #context_set_size = 1000
        self.batch_size = test_batch_size 

        # Device
        device_ = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        self.images = {}
        self.labels = {}
        
        # Number of query examples / class
        self.n_query = 10
        
        # Number of ways
        self.way_opt = 5

        # Number of support images
        self.n_support = 5

        # Cap for the support examples 
        self.cap_support_per_class = 700

        
        # # For dataset
        # for dataset in self.datasets:
        #     self.images[dataset['name']+str(dataset['task'])] = None 
        #     self.labels[dataset['name']+str(dataset['task'])] = None 
        
        # # Across all the datasets
        # for dataset in self.datasets:
            
        #     # 2 datasets need to be manually configured: sun397 and diabetic_retinopathy_detection
        #     if dataset['name'] != 'diabetic_retinopathy_detection' and dataset['name']!='sun397': #dataset['name']!='caltech101' and dataset['name']!='resisc45' and dataset['name'] != 'diabetic_retinopathy_detection':
        #         # print("Current dataset: {}".format(dataset['name']))
        #         # Dataset reader
        #         dataset_reader = TfDatasetReader(
        #                 dataset=dataset['name'],
        #                 task=dataset['task'],
        #                 context_batch_size=self.context_set_size,
        #                 target_batch_size=self.batch_size, #self.args.batch_size,
        #                 path_to_datasets='/cmlscratch/sbasu12/projects/datasets',
        #                 image_size=224, #
        #                 device='cpu' # Can be changed but cuda() not required at this point
        #             )

        #         # Context Images
        #         context_images, context_labels = dataset_reader.get_context_batch()
        #         print(f'Dataset name: {dataset}')
                
            
        #         # Test_set size
        #         test_set_size = dataset_reader.get_target_dataset_length()
                
        #         # Number of batches
        #         num_batches = self._get_number_of_batches(test_set_size)

        #         # Unique labels
        #         unique_labels = torch.unique(context_labels)
        #         print("###########################")
        #         # Total number of test images
        #         print(f'Total number of test images: {num_batches*40}')
        #         print(f'Average number of images per class: {(num_batches*40)/len(unique_labels)}')
        #         # Unique number of classes
        #         print("Current dataset: {}".format(dataset['name']))
        #         # print(f'Number of classes: {len(unique_labels)}')

        #         # Target Labels / Logits
        #         target_logits = []
        #         target_labels = []

        #         print("############################")

        #         total_images = []
        #         total_labels = []
        #         # Each batch is of size (self.batch_size)
        #         for batch in range(num_batches):
        #             batch_target_images, batch_target_labels = dataset_reader.get_target_batch()
        #             # batch_logits = self.finetune_model.predict(batch_target_images, MetaLearningState.META_TEST)
        #             # target_logits.append(batch_logits)
        #             # target_labels.append(batch_target_labels)
        #             # print(torch.unique(batch_target_labels))
        #             # print(batch_target_images.shape)

        #             # Total Images
        #             total_images.extend(batch_target_images)
        #             total_labels.extend(batch_target_labels)


        #         # total_images = torch.tensor(total_images)
        #         total_images = torch.stack(total_images)
        #         total_labels = torch.tensor(total_labels)

        #         self.images[dataset['name']+str(dataset['task'])] = total_images 
        #         self.labels[dataset['name']+str(dataset['task'])] = total_labels


                

    
    # Function to sample the VTAB dataset
    def sampler(self, dataset_name = 'caltech101', task = None):
        print("Dataset Name: {}".format(dataset_name))
        print("Task: {}".format(task))

        ##############################################################
        # Tf Dataset reader
        dataset_reader = TfDatasetReader(
                dataset=dataset_name, #['name'],
                task=task, 
                context_batch_size=self.context_set_size,
                target_batch_size=self.batch_size, #self.args.batch_size,
                path_to_datasets='/cmlscratch/sbasu12/projects/datasets',
                image_size=128, #
                device='cpu' # Can be changed but cuda() not required at this point
            )

        # Context Images
        context_images, context_labels = dataset_reader.get_context_batch()
        
        # Test_set size
        test_set_size = dataset_reader.get_target_dataset_length()
        
        # Number of batches
        num_batches = self._get_number_of_batches(test_set_size)

        # Unique labels
        unique_labels = torch.unique(context_labels)
        print("###########################")
        # Total number of test images
        print(f'Total number of test images: {num_batches*40}')
        print(f'Average number of images per class: {(num_batches*40)/len(unique_labels)}')
        # Unique number of classes
        print("Current dataset: {}".format(dataset_name))
        # print(f'Number of classes: {len(unique_labels)}')
        print(f'Number of batches: {num_batches}')


        # Target Labels / Logits
        target_logits = []
        target_labels = []

        print("############################")

        total_images = []
        total_labels = []
        counter = 0
        # Each batch is of size (self.batch_size)
        for batch in range(num_batches):
            batch_target_images, batch_target_labels = dataset_reader.get_target_batch()
            # batch_logits = self.finetune_model.predict(batch_target_images, MetaLearningState.META_TEST)
            # target_logits.append(batch_logits)
            # target_labels.append(batch_target_labels)
            # print(torch.unique(batch_target_labels))
            # print(batch_target_images.shape)

            # Total Images
            total_images.extend(batch_target_images)
            total_labels.extend(batch_target_labels)
            counter += 1

            if dataset_name == 'dsprites':
                if counter == 500:
                    break 


        # Total Pool of Images
        print(f'Total Pool for Images: {len(total_images)}')

        # total_images = torch.tensor(total_images)
        total_images = torch.stack(total_images)
        total_labels = torch.tensor(total_labels)

        ############################################################################

        # Sampler ==> 
        # total_images = self.images[dataset_name + task]
        # total_labels = self.labels[dataset_name + task] 
        
        # Unique Labels
        unique_labels = list(torch.unique(total_labels).numpy())

        # Support Images + Labels
        support_images = []
        support_labels = []
        # Query Images + Labels
        query_images = []
        query_labels = []
        
        class_main = []
        for class_ in unique_labels:
            index = torch.where(total_labels == class_)[0].numpy()

            if len(index) > 20:
                class_main.append(class_)

        # Total number of classes
        print("Total Number of Classes: {}".format(len(class_main)))

        # Number of selected classes
        selected_classes = random.sample(class_main, self.way_opt)
        print("Selected Classes: {}".format(selected_classes))


        # For class_id in selected classes
        for class_id in selected_classes:
            # Get the overall index for all class
            indexes = torch.where(total_labels == class_id)[0].numpy()
            
            # Extracted Image
            curr_images = total_images[indexes]
            
            # Hit only if number of images is less than the number of query
            if len(curr_images) <= self.n_query:
                query_images_ = curr_images[:int(self.n_query/2)]
                support_images_ = curr_images[int(self.n_query/2):]

            else:
                query_images_ = curr_images[:self.n_query]
                # Pool of support images --- Set a cap per class --- By default the cap is set as 400.
                support_images_ = curr_images[self.n_query:][:self.cap_support_per_class]


            # Query + Support Labels
            query_labels_ = [selected_classes.index(class_id) for i in range(0, len(query_images_))]
            support_labels_ = [selected_classes.index(class_id) for i in range(0, len(support_images_))]

            # Support + Query Images
            support_images.extend(support_images_)
            query_images.extend(query_images_)

            # Support + Query Labels
            support_labels += support_labels_
            query_labels += query_labels_


        ##########################################################################################
        # Support Images, query images, support labels, query_labels
        support_images = torch.unsqueeze(torch.stack(support_images), dim = 0)
        query_images = torch.unsqueeze(torch.stack(query_images), dim=0)
        support_labels = torch.tensor(support_labels).reshape(1,-1)
        query_labels = torch.tensor(query_labels).reshape(1,-1)
        

        ##########################################################################################
        
        return support_images, support_labels, query_images, query_labels


    # Number of batches
    def _get_number_of_batches(self, task_size):
        num_batches = int(np.ceil(float(task_size) / float(self.batch_size)))
        if num_batches > 1 and (task_size % self.batch_size == 1):
            num_batches -= 1

        return num_batches


# import tensorflow as tf 
# import tensorflow_datasets as tfds

# {'name': "caltech101", 'task': None, 'enabled': True}, # 3060 examples 
#             {'name': "cifar100", 'task': None, 'enabled': True}, # 
#             {'name': "oxford_flowers102", 'task': None, 'enabled': True},
#             {'name': "oxford_iiit_pet", 'task': None, 'enabled': True},
#             {'name': "sun397", 'task': None, 'enabled': True},
#             {'name': "svhn_cropped", 'task': None, 'enabled': True},
#             {'name': "eurosat", 'task': None, 'enabled': True},
#             {'name': "resisc45", 'task': None, 'enabled': True},
#             {'name': "patch_camelyon", 'task': None, 'enabled': True},
#             {'name': "diabetic_retinopathy_detection", 'task': None, 'enabled': True},
#             {'name': "clevr", 'task': "count", 'enabled': True},
#             {'name': "clevr", 'task': "distance", 'enabled': True},
#             {'name': "dsprites", 'task': "location", 'enabled': True},
#             {'name': "dsprites", 'task': "orientation", 'enabled': True},
#             {'name': "smallnorb", 'task': "azimuth", 'enabled': True},
#             {'name': "smallnorb", 'task': "elevation", 'enabled': True},
#             {'name': "dmlab", 'task': None, 'enabled': True},
#             {'name': "kitti", 'task': None, 'enabled': True},

# names = ['caltech101', 'cifar100', 'oxford_flowers102', 'oxford_iiit_pet', 'svhn_cropped', 'eurosat', 'resisc45', 'patch_camelyon', 'clevr', 'clevr', 'dsprites', 'dsprites', 'smallnorb', 'smallnorb', 'dmlab', 'kitti']            
# tasks = [None, None, None, None, None, None, None, None, 'count', 'distance', 'location', 'orientation', 'azimuth', 'elevation', None, None]


# # # vtab 
vtab = VTAB('/home', context_set_size = 1000)
# vtab.sampler()





# Caltech 101 --> 3060 examples
# import ssl  
# import os 
# # os.environ['http_proxy'] = 'http://127.0.0.1:12333'  
# # os.environ['https_proxy'] = 'https://127.0.0.1:12333'  
# # ssl._create_default_https_context = ssl._create_unverified_context

# from tensorflow_datasets.core.utils import gcs_utils
# gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
# gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False


# print("Testing")
# dataset, info = tfds.load(name='cifar100', split='train', with_info=True, data_dir = '/home/t-sambasu/intern/PMF/metadataset_pmf/vtab_datasets', try_gcs=False)


# # #assert tfds.core.utils.gcs_utils.exists('gs://tfds-data/dataset_info/coco/2014/1.1.0') == False
# # print(info)
# #a = tfds.list_builders()
# #print(a)