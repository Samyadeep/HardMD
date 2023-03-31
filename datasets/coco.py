""" Data-loader for COCO for few-shot classification
- Has the sampler which creates query which is randomly sampled amongst the images with less objects, 

"""


# Libraries
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
import torchvision.transforms as transforms
import json 
import random 
import torch 
import numpy as np 
import os 

# Data transform borrowed from objectnet
class data_transform:
    def __init__(self):
        self.model_pretrain_params = {}
        self.model_pretrain_params['input_size'] = [3, 224, 224]
        self.model_pretrain_params['mean'] = [0.485, 0.456, 0.406] # Imagenet configurations
        self.model_pretrain_params['std'] = [0.229, 0.224, 0.225] # Imagenet configurations
        self.resize_dim = 128 #self.model_pretrain_params['input_size'][1]


    def getTransform(self):
        trans = transforms.Compose([transforms.Resize((self.resize_dim, self.resize_dim)), # Resize to 224, 224
                                    #transforms.CenterCrop(self.model_pretrain_params['input_size'][1:3]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.model_pretrain_params['mean'],
                                                         std=self.model_pretrain_params['std'])
                                    ])

        # 
        return trans


# COCO dataset
COCO_dataset = '/hdd/t-sambasu/train2017/train2017'
COCO_ann = '/hdd/t-sambasu/annotations_trainval2017/annotations/instances_train2017.json'


# ObjectNet dataset
class COCOVision(VisionDataset):
    """
    ObjectNet dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
    """

    def __init__(self, root, annfile):
        """Init ObjectNet pytorch dataloader."""
        super(COCOVision, self).__init__(root, annfile)

        # COCO Tools
        from pycocotools.coco import COCO
        transform = None 
        target_transform = None 

        # Attributes
        self.root = root 
        self.coco = COCO(annfile)
        
        # Self.ids
        self.ids = list(self.coco.imgs.keys())
        #self.transform = transform
        self.target_transform = target_transform

        # Transformation function
        self.transform = data_transform().getTransform()

        # PIL_Loader
        self.loader = self.pil_loader

        ################## Modify for use-case of search ###################### 
        # Stores the path of images
        self.paths =  []
        print(f'Total number of images: {len(self.ids)}')
        # Store the path of labels
        self.labels = []
        
        # Across image_ids ===> 
        for img_id in self.ids:
            self.paths.append(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name']))
            ann_ids = self.coco.getAnnIds(imgIds = img_id)

            # Comes with segmentation masks, bounding box coordinates, image_classes (Segmentation classes)
            target = self.coco.loadAnns(ann_ids)
            curr_label = [segment['category_id'] for segment in target]
            self.labels.append(curr_label)

        
        #######################################################################
        #######################################################################
        
        # Number of query examples
        self.q = 10
        # Number of ways
        self.ways = 5
        # Number of images / class which should have a single object per image at a minimum
        self.min_images = 30
        ##### Pre-process classes where the number of images are greater than self.min_images

        # Support Pool cap per class
        self.support_memory_cap = 500

        # Unique classes
        unique_classes = []
        for clss in self.labels:
            unique_classes += clss 

        # Filtered classes
        self.filtered_classes = []

        # Total number of classes 
        unique_classes = list(set(unique_classes))
        print(f'Total number of classes: {len(unique_classes)}')

        # For every class count the number of images which have a single object --- set with a threshold
        for cl in unique_classes:
            # Number of objects
            num_objects = []
            num_single_objects = 0
            
            # Go through each image
            for label in self.labels:
                if cl in label:
                    if len(label) == 1:
                        num_single_objects += 1


            # If number of images with single objects is greater than a threshold
            if num_single_objects >=self.min_images:
                self.filtered_classes.append(cl)
        
        # Total number of available classes
        print(f'Total number of classes available : {len(self.filtered_classes)}')

    
    # PIL Loader
    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    # Sampler for the Support and Query sets with rules about selection
    def sampler(self):
        # Sample classes / ways
        unique_classes = random.sample(self.filtered_classes, self.ways)
        ###########################################################################

        # Sample the query set 
        query_indexes = []
        query_labels = []

        support_indexes = []
        support_labels = []

        count = 0
        # Walk through the selected classes and create the query index and support index
        for cl in unique_classes:
            # Current index: 0
            index = 0
            query_indexes_ = []

            for label in self.labels:
                if cl in label:
                    # Choose for query index: as this image has only one object
                    if len(label) == 1:
                        num_single_objects = 1

                        # Add to the query index
                        query_indexes_.append(index)
                    
                    # Add every thing to support Index
                    support_indexes.append(index)
                    support_labels.append(cl)
                        
                
                index += 1
            
            # Randomly shuffle
            random.shuffle(query_indexes_)
            query_indexes_selected = random.sample(query_indexes_, self.q)
            query_indexes += query_indexes_selected
            query_labels += [cl for i in range(0, self.q)] 


        # Refine query indexes -- Subselect a random set
        support_indexes_ = []
        support_labels_ = []

        ######## Filter based on the ones which are not present in query ########
        print(f'Original length of support: {len(support_indexes)}')
        for i in range(0, len(support_indexes)):
            if support_indexes[i] not in query_indexes:
                support_indexes_.append(support_indexes[i])
                support_labels_.append(support_labels[i])
        

        # Filtered length of support 
        print(f'Filtered length of support: {len(support_indexes_)}')

        print("##################### Starting the process of filtering for setting a cap per class #########################")        
        # TODO : Add an argument for adding a memory cap for support pool (Add the memory cap per class)
        # Filtering 
        support_indexes_final = []
        support_labels_final = []
        print(f'Number of unique classes: {len(unique_classes)}')
        
        # cls_id
        for cls_id in unique_classes:
            # Positions
            positions_ = np.where(np.array(support_labels_) == cls_id)[0]
            # 
            random.shuffle(positions_)
            curr_positions = positions_[:self.support_memory_cap]

            for pos_ in curr_positions:
                support_indexes_final.append(support_indexes_[pos_])
                support_labels_final.append(support_labels_[pos_])

            
        # Final set of support_set
        print(f'Final length of support pool data : {len(support_indexes_final)}')
        print(f'Final length of support labels: {len(support_labels_final)}')
        # Final set of query_set
        print(f'Final length of query set data: {len(query_indexes)}')
        print(f'Final length of query label data: {len(query_labels)}')

        # Extract the images from the given support indexes
        support_images = []
        query_images = []
        support_labels = []
        

        # Going through support images
        for sup_index in support_indexes_final:
            curr_image = self.transform(self.loader(self.paths[sup_index]))
            support_images.append(curr_image)

            
        # Going through query images
        for query_index in query_indexes:
            curr_image = self.transform(self.loader(self.paths[query_index]))
            query_images.append(curr_image)

        # Support Labels + Query Labels    
        support_labels_ = [unique_classes.index(label) for label in support_labels_final]
        query_labels_ = [unique_classes.index(label) for label in query_labels]

        # Support Images
        support_images = torch.unsqueeze(torch.stack(support_images), dim = 0)
        query_images = torch.unsqueeze(torch.stack(query_images), dim = 0)
        support_labels = torch.tensor(support_labels_).reshape(1,-1)
        query_labels = torch.tensor(query_labels_).reshape(1,-1)  
        
        ################################################### 
        # support_indexes_final --- to_return 
        # query_indexes --- to_return
        ###################################################
        
        
        return support_images, support_labels, query_images, query_labels, support_indexes_final, query_indexes


    # Length of dataset
    def __len__(self):
        return len(self.ids)
    
    
    # #  Get__item()
    # def __getitem__(self, index):
    #     coco = self.coco 
    #     img_id = self.ids[index]
    #     ann_ids = coco.getAnnIds(imgIds = img_id)

    #     # Comes with segmentation masks, bounding box coordinates, image_classes (Segmentation classes)
    #     target = coco.loadAnns(ann_ids)
    #     # Category IDs
    #     category_ids = []

    #     # Path
    #     path = coco.loadImgs(img_id)[0]['file_name']

    #     # Append the category ID
    #     for segment in target:
    #         category_ids.append(segment['category_id'])
        
    #     return 
    

   

# # COCO dataset loader
# c = COCOVision(COCO_dataset, COCO_ann)
# c.sampler()

