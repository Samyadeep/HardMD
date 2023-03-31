""" Objectnet data-loader for the optimization objective """

# Libraries
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
import torchvision.transforms as transforms
import json 
import random 
import torch 


# Mapping files
# mapping_file = "/home/t-sambasu/intern/PMF/metadataset_pmf/datasets/imagenet_pytorch_id_to_objectnetid.json"
# with open(mapping_file,"r") as f:
#     mapping = json.load(f)
#     # convert string keys to ints
#     mapping = {int(k): v for k, v in mapping.items()}

# print("Mapping between Imagenet ID ====> ObjectNet ID : {}".format(len(mapping)))


# Data transform
class data_transform:
    def __init__(self):
        self.model_pretrain_params = {}
        self.model_pretrain_params['input_size'] = [3, 224, 224]
        self.model_pretrain_params['mean'] = [0.485, 0.456, 0.406]
        self.model_pretrain_params['std'] = [0.229, 0.224, 0.225]
        self.resize_dim = self.model_pretrain_params['input_size'][1]

    def getTransform(self):
        trans = transforms.Compose([transforms.Resize((self.resize_dim, self.resize_dim)),
                                    #transforms.CenterCrop(self.model_pretrain_params['input_size'][1:3]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.model_pretrain_params['mean'],
                                                         std=self.model_pretrain_params['std'])
                                    ])
        return trans


# ObjectNet dataset
class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.ToTensor'
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        img_format (string): jpg
                             png - the original ObjectNet images are in png format
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None, img_format="jpg"):
        """Init ObjectNet pytorch dataloader."""
        super(ObjectNetDataset, self).__init__(root, transforms, transform, target_transform)
        
        # 
        self.loader = self.pil_loader
        self.img_format = img_format
        files = glob.glob(root+"/**/*."+img_format, recursive=True)
        # classes = glob.glob(root + "/images/*", recursive = True)
        # print(classes)

        # Path dictionary
        self.pathDict = {}

        # Class mappings
        self.class_mappings = {}
        for f in files:
            self.pathDict[f.split("/")[-1]] = f
            if f.split("/")[-2] in self.class_mappings:
                self.class_mappings[f.split("/")[-2]].append(f)
            else:
                # Class mappings
                self.class_mappings[f.split("/")[-2]] = []

        # Set of images
        self.imgs = list(self.pathDict.keys())
        

    """ 
    Uses the class_mappings dictionary to construct the episodic set
    - Returns a dictionary with {class: path_of_images}
    
    """
    def sampler(self, args):
        # Sample classes 
        classes = list(self.class_mappings.keys())
        
        # Total number of classes
        # print(f'Number of classes: {len(classes)}')

        # Number of classes to select
        num_classes = args.way_opt

        # Episode classes
        episode_classes = random.sample(classes, num_classes)
        
        # Class dictionary
        class_dict = {k: v for k, v in self.class_mappings.items() if k in episode_classes}

        return class_dict


    """ 
    Function to convert to images
    - takes in support pool, query_images, support_labels, query_labels
    - returns the image array
    """ 
    def convert_to_images(self, support_pool_path, support_pool_labels, query_images_path, query_labels, class_keys):
        # Support Images/Labels
        support_images = []
        support_labels = []

        # Query Images/Labels
        query_images_ = []
        query_labels_ = []

        ################### Create the support images / labels ########################
        c = 0
        for sup_img_path in support_pool_path:
            # load the image from the path
            img = self.loader(sup_img_path)
            if self.transforms is not None:
                img, _ = self.transforms(img, img)
            # Support images appending
            support_images.append(img)
            # Load the labels
            support_labels.append(class_keys.index(support_pool_labels[c]))
            c += 1
            
        
        #################### Create the query images / labels #########################
        c = 0
        for query_img_path in query_images_path: 
            # Load the image from the path
            img = self.loader(query_img_path)
            if self.transforms is not None:
                img, _ = self.transforms(img, img)
            # Query images
            query_images_.append(img)
            # Query labels append
            query_labels_.append(class_keys.index(query_labels[c]))
            c += 1

        ################################################################################
        # Create the torch tensor for the support images 
        support_images = torch.unsqueeze(torch.stack(support_images), dim = 0)

        # Create the torch tensor the query images
        query_images = torch.unsqueeze(torch.stack(query_images_), dim = 0)
        
        # Support / Query labels
        support_labels = torch.tensor(support_labels).reshape(1,-1)
        query_labels = torch.tensor(query_labels_).reshape(1,-1)

        # 
        return support_images, support_labels, query_images, query_labels, sup_img_path, query_img_path
    

    
    # Original __getitem__() function
    def __getitem__(self, index):
        """
        Get an image and its label.
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the image file name
        """
        img, target = self.getImage(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def getImage(self, index):
        """
        Load the image and its label.
        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
        img = self.loader(self.pathDict[self.imgs[index]])

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)
        return (img, self.imgs[index])

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
