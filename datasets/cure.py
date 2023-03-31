"""
Data-loader for CURE-OR dataset for few-shot evaluation: 

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
from PIL import Image


# CURE dataset
CURE_dataset = '/hdd/t-sambasu/01_no_challenge'

# "backgroundID_deviceID_objectOrientationID_objectID_challengeType_challengeLevel.jpg"


# Data transform borrowed from objectnet
class data_transform:
    def __init__(self):
        self.model_pretrain_params = {}
        self.model_pretrain_params['input_size'] = [3, 224, 224]
        self.model_pretrain_params['mean'] = [0.485, 0.456, 0.406]
        self.model_pretrain_params['std'] = [0.229, 0.224, 0.225]
        self.resize_dim = self.model_pretrain_params['input_size'][1]

    def getTransform(self):
        trans = transforms.Compose([transforms.Resize((self.resize_dim, self.resize_dim)), # Resize to 224, 224
                                    #transforms.CenterCrop(self.model_pretrain_params['input_size'][1:3]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.model_pretrain_params['mean'],
                                                         std=self.model_pretrain_params['std'])
                                    ])
        return trans



# Dataset 
class CUREVision(VisionDataset):
    """
    CURE dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
    """
    def __init__(self, root):
        """Init CURE-OR dataloader."""
        super(CUREVision, self).__init__(root)

        self.root = root 
        # Backgrounds
        backgrounds = ['3d1', '3d2', 'texture1', 'texture2', 'white']
        # Load the paths of images 
        devices = ['DSLR_JPG', 'HTC', 'iPhone', 'LG', 'Logitech']

        # Total paths
        paths = []

        # Backgrounds + Devices
        for back in backgrounds:
            for dev in devices:
                curr_path = os.path.join(self.root, back, dev)

                # Extract all the images of the folder
                files = glob.glob(curr_path + '/*.jpg', recursive=True)
                paths += files 
        
        # Filter the paths with (object_id[class], background_id, rotation(orientation)_id)
        labels, rotations, backgrounds, device_ids = self.filter(paths)

        # Assign to self. variables
        self.paths = paths 
        self.labels = labels 
        self.rotations = rotations 
        self.backgrounds = backgrounds 
        self.device_ids = device_ids 

        # Variables for selecting the episode
        self.q = 10
        self.way = 5

        self.loader = self.pil_loader
        # Self.transform
        self.transform = data_transform().getTransform()

    
    # Function to filter the paths
    def filter(self, paths):
        # Filtered paths 
        filtered_paths = []
        labels = []
        rotations = []
        backgrounds = []
        device_ids = []


        # Path information
        for path in paths:
            # Current Path
            curr_path = path.split("/")[-1]
            # "backgroundID_deviceID_objectOrientationID_objectID_challengeType_challengeLevel.jpg"
            # 
            ids = curr_path.split('_')
            background_id = ids[0]
            device_id = ids[1]
            orientation_id = ids[2]
            objectId = ids[3] # Label

            labels.append(objectId)
            rotations.append(orientation_id)
            backgrounds.append(background_id)
            device_ids.append(device_id)

           
        return labels, rotations, backgrounds, device_ids

    
    # PIL Loader
    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    # Sampler for episodes / tasks
    def sampler(self):
        # Sample ways   
        unique_labels = list(set(self.labels))
        random.shuffle(unique_labels)

        
        # Selected classes
        selected_classes = random.sample(unique_labels, self.way)

        # Support Indexes
        support_indexes = []
        support_labels = []

        # Query Indexes
        query_indexes = []
        query_labels = []

        # For every class ----> Sample query set and support pool
        for class_id in selected_classes:
            # Sample query set 
            image_indexes = list(np.where(np.array(self.labels) == class_id)[0])
            image_indexes_rotation_1 = [self.rotations[index] for index in image_indexes]
            # Index zipping
            index_zip = list(zip(image_indexes, image_indexes_rotation_1))
            
            # Total query indexes
            query_index = [zip_index[0] for zip_index in index_zip] #if zip_index[1] == '1'] 
            random.shuffle(query_index)

            # Query Indexes 
            query_index_ = random.sample(query_index, self.q)

            for q_index in query_index_:
                query_indexes.append(q_index)
                query_labels.append(class_id)
            

            for img_index in image_indexes:
                if img_index not in query_index_:
                    support_indexes.append(img_index)
                    support_labels.append(class_id)
            

        ########################### Support Pool + Query set Creation Complete ###########################
        support_paths = [self.paths[id_] for id_ in support_indexes]
        query_paths = [self.paths[id_] for id_ in query_indexes]

        # Support Orientations
        support_orientations = [self.rotations[id_] for id_ in support_indexes]
        query_orientations = [self.rotations[id_] for id_ in query_indexes]

        ############################ Reading Images #############################
        # Support Image / Query Images
        support_images = [] 
        query_images = []   

        # Concatenate support paths ===> Images
        for path in support_paths:
            # Read from the path 
            if self.transform is not None:
                image_ = self.transform(self.loader(path))
                support_images.append(image_)
        
        # Concatenate query paths ===> Images 
        for path in query_paths:
            if self.transform is not None:
                image_ = self.transform(self.loader(path))
                query_images.append(image_)
        

        # Support + Query Images
        support_images = torch.unsqueeze(torch.stack(support_images), dim = 0)
        query_images = torch.unsqueeze(torch.stack(query_images), dim = 0)

        ##################################################################################################
        # Convert the labels
        support_classes = []
        query_classes = []

        for cl_ in support_labels:
            support_classes.append(selected_classes.index(cl_))
        
        for cl_ in query_labels:
            query_classes.append(selected_classes.index(cl_))
        
        # Convert to torch tensor 
        support_labels = torch.tensor(support_classes).reshape(1,-1)
        query_labels = torch.tensor(query_classes).reshape(1,-1)
        
        ##################################################################################################
        # Support Orientations, Query_orientations only contain the rotations_id
        return support_images, support_labels, query_images, query_labels, support_orientations, query_orientations
    

# Cure vision dataset
# c = CUREVision('/hdd/t-sambasu/01_no_challenge')
# # Sampler which returns the support pool and query images / labels
# support_images, support_labels, query_images, query_labels, support_orientations, query_orientations = c.sampler()

