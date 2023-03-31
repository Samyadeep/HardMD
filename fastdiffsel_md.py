""" 
Implementation of FastDiffSel from 'Hard-Meta-Dataset: Towards Understanding Few-Shot Performance on Difficult Tasks' (ICLR 2023)


"""


# Libraries
import sys
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from engine import train_one_epoch, evaluate
import utils.deit_util as utils
from datasets import get_loaders
from utils.args import get_args_parser
from models import get_model


# Importing the classes for dataset creation
from datasets.meta_dataset import config as config_lib
from datasets.meta_dataset import sampling
from datasets.meta_dataset.utils import Split
from datasets.meta_dataset.transform import get_transforms
from datasets.meta_dataset import dataset_spec as dataset_spec_lib

# Import Os
import os 
import pickle


# Other dependencies
import h5py
from PIL import Image
import json
#import cv2

from models import get_model
from datasets.meta_dataset import config as config_lib
import torchvision.transforms as transforms

import time 
# Pull the accuracy
from timm.utils import accuracy, ModelEma
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

# Normalize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



"""Transform function  - Test
Args:
 - data_config: Data config from the original TF-MD codebase 
Note: In case of test only resize, center_crop, to_tensor and normalize is done
""" 
def test_transform(data_config):
    resize_size = int(data_config.image_size * 256 / 224)
    assert resize_size == data_config.image_size * 256 // 224
    # resize_size = data_config.image_size
    
    transf_dict = {'resize': transforms.Resize(resize_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size), # This will result in a 128x128 Crop
                   'to_tensor': transforms.ToTensor(), # Convert to tensor
                   'normalize': normalize} # Normalize
    augmentations = data_config.test_transforms
    
    return transforms.Compose([transf_dict[key] for key in augmentations])






""" Projection Step for sparsity
args: Arguments 
curr_weights: Current weights for each example in the prototype
Kmax: Maximum sparsity value
"""
def projection_step(args, curr_weights, Kmax):
    # Detach the weights from the computational graph
    #curr_weights_np = curr_weights.detach().cpu().numpy()
    curr_weights_np = curr_weights[0]
    
    # Functional value of the dual gradient (c -> dual functional)
    def f_(curr_weight, c):
        # Compute function value 
        f_abs = np.abs(curr_weight) - c
        f_abs[f_abs <=0] = 0
        # Dual gradient functional
        dual_gradient_functional = np.sum(f_abs) - Kmax # Kmax is a global variable
        ########## End functional

        return dual_gradient_functional

    
    """ Bisection method to find the roots for the optimal dual parameter """
    def compute_dual_gradient(curr_weight, lambda_start, lambda_end):
        # Lambda dual 
        lambda_dual = (lambda_start + lambda_end)/2
        # print("Starting value of the bisection method: {}".format(lambda_start))
        # print("Max value of the parameter : {}".format(lambda_end))
        # print(lambda_dual)

        # Number of iterations
        NMAX = 20
        TOL = 0.0001

        optimal_lambda = None 

        for i in range(0, NMAX):
            # Mid point
            lambda_dual = (lambda_start + lambda_end)/2

            ####### Functional 
            dual_gradient_functional = f_(curr_weight, lambda_dual)
            if dual_gradient_functional == 0  or (lambda_end - lambda_start)/2 < TOL:
                optimal_lambda = lambda_dual
                break 

            # 
            f_c = f_(curr_weight, lambda_dual)
            f_a = f_(curr_weight, lambda_start)

            # Same sign
            if (f_a >= 0 and f_c >= 0) or (f_a <=0 and f_c <=0):
                lambda_start = lambda_dual 
            
            else:
                lambda_end = lambda_dual

            # Functional values
            #print("Functional value: {}".format(f_(curr_weight, lambda_dual)))

        if optimal_lambda == None:
            optimal_lambda = lambda_dual
        
        #print("Functional value: {}".format(f_(curr_weight, lambda_dual)))
        #print("Lambda dual: {}".format(lambda_dual))
        return optimal_lambda
    

    # Range for bisection search
    lambda_start = 0
    lambda_end = np.max(np.abs(curr_weights_np)) # Changed (bounds / range for the dual variable)

    # Dual parameters (\lambda)
    dual_param = compute_dual_gradient(curr_weights_np, lambda_start, lambda_end)
    
    # Compute the optimal projected weights for the examples -- obtained by the proximal operator
    sign_weights = np.sign(curr_weights_np)
    
    # proximal_update (|z| - dual_optimal_param)
    proximal_update = np.abs(curr_weights_np) - dual_param 
    proximal_update[proximal_update<=0] = 0

    # Current weights 
    curr_weights_np = np.multiply(sign_weights, proximal_update)

    # Number of positions where the weights are non-zero
    non_zero_weights = len(np.where(curr_weights_np >0)[0])

    # Update the torch weights with the numpy array which has the projected weights
    #temp = torch.tensor(curr_weights_np.reshape(1,-1), requires_grad=True, device=args.device)

    return curr_weights_np




# Function for computing the 
def compute_hard_ep_acc(args, model, support_images, support_labels, query_images, query_labels, support_indexes_total, class_keys, class_h5_dict, base_source, k, num_support_to_sample, save=False):
    criterion = torch.nn.CrossEntropyLoss()
    """ Extracting the hard episodes from the weights """
    # Extract the weights which are high to corresponding to computing the prototypes
    curr_training_weights = model.weights.detach().cpu().numpy()[0]

    # Number of support_images
    support_images_np = support_images.detach().cpu().numpy()[0]
    # Support Labels Numpy
    support_labels_np = support_labels.detach().cpu().numpy()[0]

    
    # Unique labels
    labels_unique = np.unique(support_labels_np)
    

    # Populate this with the number of support per class
    sampling_supports = {}

    for l in num_support_to_sample:
        sampling_supports[l[0]] = l[1]

    #print(sampling_supports)
    
    way_curr = len(labels_unique)
    
    # Worst support examples
    hard_support_indexes = []
    
    save_path = '/home/t-sambasu/intern/PMF/metadataset_pmf/optimization_vis/' + base_source
    num_images_per_class = {}

    for label in labels_unique:
        if label not in num_images_per_class:
            num_images_per_class[label] = 0


        # Positions where the given support examples are present
        pos_label = np.where(support_labels_np == label)[0]

        # Extract the weights
        class_weights = curr_training_weights[pos_label]
        
        # Positions which are non-zero
        pos_non_zero = np.where(class_weights > 0)[0]
        
        # Check the sparsity ratio
        pos_zero = np.where(class_weights == 0)[0]
        
        #print(f'Number of zero elements is {len(pos_zero)} out of {len(pos_label)} for label : {label}')
        #print(class_weights)
        # Sorted indexes of the weights from higher to lower
        if args.md_sampling == False:
            #print("Sampling Fixed-shot")
            sorted_weight_indexes = np.argsort(class_weights)[::-1][:args.sup]
        
        else:
            #print("Sampling Variable-Shot")
            sorted_weight_indexes = np.argsort(class_weights)[::-1][:sampling_supports[label]]

        # Current class 
        if len(pos_non_zero) == 0:
            pos_non_zero = np.array([0,1])
        
        # Extraction of the set 
        support_indexes = pos_label[sorted_weight_indexes] #pos_label[pos_non_zero] 

        # Hard support indexes
        hard_support_indexes += list(support_indexes)
        num_images_per_class[label] = len(support_indexes)

        ####################### Get the support images #########################

        # if save == True:
        #     # Current Class
        #     curr_class = class_keys[label]
        #     # print(type(support_indexes_total))
        #     # print(list(support_indexes))
        #     current_support_hard_images = [support_indexes_total[i] for i in support_indexes]

        #     h5_file = class_h5_dict[base_source][curr_class]
        #     c = 0
        #     for index in current_support_hard_images:
        #         record = h5_file[index]

        #         # Record
        #         x = record['image'][()]
        #         x = Image.fromarray(x)

        #         # Save path
        #         x.save(save_path + '/support_' + str(curr_class) + '_iteration_' + str(k) + '_' +  str(c) + '.png')
        #         c += 1
            

    # print("#################### Computing the final accuracy for the hard episode ######################")
    # # Remove the weighted prototype scheme
    
    model.weights = None 
    model.class_wise_weights = None
    
    #weight_indexes = np.where(curr_training_weights > 0)[0]
    #print("Number of hard examples in the episode: {}".format(len(hard_support_indexes)))

    # # Support images which are hard
    support_images_hard = torch.unsqueeze(torch.tensor(support_images_np[hard_support_indexes]), dim=0).to(args.device) #torch.unsqueeze(torch.tensor(support_images_np[hard_support_indexes]), dim=0).to(args.device)
    # Support labels which are hard
    support_labels_hard = torch.unsqueeze(torch.tensor(support_labels_np[hard_support_indexes]), dim=0).to(args.device) #torch.unsqueeze(torch.tensor(support_labels_np[hard_support_indexes]), dim=0).to(args.device)
    

    # Don't create computational graph to save memory
    with torch.no_grad():
        output = model.forward_light(support_images_hard, support_labels_hard, query_images)
    

    # Output 
    output = output.view(query_images.shape[0] * query_images.shape[1], -1)
    
    # Compute the loss 
    query_labels = query_labels.view(-1)
    loss = criterion(output, query_labels)
    loss_value = loss.item()

    # accuracy of the hard episode
    acc1_hard_episode, acc5 = accuracy(output, query_labels, topk=(1, 5))
    cl_wise_acc,total = class_wise_acc(output, query_labels, way_curr)
    
    # print("Hard episode Acc Final: {}".format(acc1_hard_episode))
    # print("Hard episode Class-wise accuracy: {}".format(cl_wise_acc))
    # print("Acc hard: {}".format(total))
    # print("Number of images per class: {}".format(num_images_per_class))

    return acc1_hard_episode, cl_wise_acc, hard_support_indexes




""" Definition of the class-wise accuracy of an episode
Output: predictions by the model
query_labels: Ground-truth value  
"""
def class_wise_acc(output, query_labels, ways):
    _, pred = output.topk(1, 1, True, True)
    pred_ = pred.reshape(-1,)

    # Class-wise accuracy
    class_wise_acc = {}#{0:[], 1:[], 2:[], 3:[], 4:[]}   
    for j in range(0, ways):
        class_wise_acc[j] = []

    c = 0
    for label in query_labels:
        if label.item() == pred_[c]:
            class_wise_acc[label.item()].append(1)
        
        else:
            class_wise_acc[label.item()].append(0)
        
        c += 1

    # Post process
    total = []
    class_acc = []
    for i in range(0, ways):
        curr_ = class_wise_acc[i]
        total += curr_
        acc_class = sum(curr_)/len(curr_)
        #print("Class Acc: {}".format(acc_class))
        class_acc.append(acc_class)
    
    return class_acc, sum(total)/len(total)



""" 
query_images: Query image set
query_labels: Query labels
episode_description: Class wise tuple of the sampled episode
class_map: Mapping classes to the h5 records
class_h5_dict: H5 opened files 
support_pool: Data Indexes for each class from where the sampling can be done 
model: Underlying model used to extract the hard support sets
transform: Transformation function for the torch tensors
"""

# Function to run the optimization algorithm for generating the hard episodes
def generate_hard_support_sets_optimization(args, query_images, query_labels, episode_description, class_map, class_h5_dict, support_pool, model, transform, k, total_num_supports):
    # Number of Support Shot
    nb_support = args.sup 
    ways = len(list(set(query_labels)))

    # Number of ways
    print("Number of ways: {}".format(ways))

    # Get the original support set
    class_keys = list(support_pool.keys())

    # Number of support to sample (class_key, num_support_to_sample)
    num_support_to_sample = [(class_keys.index(tup[0]), tup[1]) for tup in total_num_supports]

    
    """ Store the pool of support images to search from ..... """
    # Number of Support To Sample
    # Store the losses across different epoch
    losses_total = []
    accuracies_total = []   

    # Step 1: Accumulate all the support images 
    support_images = []
    support_labels = []

    # Base source
    base_source = args.base_sources[0]
    # Total support indexes
    support_indexes_total = []

    # Original Support Images 
    support_original_images = []
    # Original Support Labels 
    support_original_labels = []

    support_transformed = []

    # cl_key
    for cl_key in class_keys:
        # Current Pool
        curr_pool = support_pool[cl_key]

        # Extract the h5 location for the current class
        h5_file = class_h5_dict[base_source][cl_key]

        # Accumulate the support images 
        for sup_key in curr_pool:
            # h5 record
            record = h5_file[sup_key]
            x = record['image'][()]

            # Read the image 
            x = Image.fromarray(x)

            v = transform(x)
            # Support Images and Labels
            support_images.append(v)
            support_transformed.append(v)

            support_labels.append(class_keys.index(cl_key))
            
            # This key can be used along with the h5 file to generate the hard support sets
            support_indexes_total.append(sup_key)


            ########### Add the original images ########
            support_original_images.append(x)
            support_original_labels.append(cl_key)

    
    # All the support images    
    support_images = torch.unsqueeze(torch.stack(support_images, dim = 0), dim=0)
    support_labels = torch.tensor(support_labels).reshape(1,-1)
    query_images = torch.unsqueeze(torch.stack(query_images, dim=0), dim=0)
    q_labels = []

    # For each of the query label -- store the right indexes
    for q_label in query_labels:
        q_labels.append(class_keys.index(q_label))

    # Query labels
    q_labels = torch.tensor(q_labels).reshape(1,-1)
    
    # Sanity Check
    print(f'Shape of Support Images: {support_images.shape}')
    print(f'Shape of Support Labels: {support_labels.shape}')
    print(f'Shape of Query Images: {query_images.shape}')
    print(f'Shape of Query Labels: {q_labels.shape}')

    # Starting the optimization process
    print("Starting the optimization procedure .....")
    
    # Pass the support images through the embedder and store them in batches 
    ########################################################################
    # The embedder is model.backbone 
    sup_images = []
    q_images = []
    
    # Batch_size for the embeddings #
    batch_size = 100
    print(f'Shape of support images: {support_images.shape}')
    for i in range(0, len(support_images[0]), batch_size):
        with torch.no_grad():
            sup_embeddings = model.backbone(support_images[0][i:i+batch_size].to(args.device))
            sup_images += sup_embeddings

        with torch.no_grad():
            query_embeddings = model.backbone(query_images[0][i:i+batch_size].to(args.device))
            #q_images.append(query_embeddings)
            q_images += query_embeddings
       

    # Support Images / Query Images - embeddings ---- These embeddings would be into cuda() memory automatically;
    support_images = torch.unsqueeze(torch.stack(sup_images, dim = 0), dim = 0) # Shape: [Num_support x embedding_size]
    query_images = torch.unsqueeze(torch.stack(q_images, dim = 0), dim = 0) # Shape: [Num_query x embedding_size]

    # 
    #support_images = support_images.to(args.device) #cuda() # Changed
    support_labels = support_labels.to(args.device) #.cuda()
    #query_images = query_images.to(args.device) #cuda() # Changed
    query_labels = q_labels.to(args.device) #.cuda() 


    # Total number of images to choose from
    print("Total Support pool to choose from: {}".format(len(support_images[0])))

    
    # Define the weight variable for choosing the images
    w = torch.ones((1, len(support_images[0])), requires_grad=True, device = args.device) #torch.rand((1, len(support_images[0])), requires_grad = True, device = args.device)
    
    # Initialise the weights for the training examples
    model.weights = w 
  
    # Debug step: Shape of the weight vector
    # print(f'The shape of the weight vector : {model.weights.shape}')
    
    # Step 2: Pass the support images through the network to get the loss
    n_epochs = args.optimizer_epoch

    # Step size for updating the weights of the training examples
    step_size = args.weighted_step_size
    
    # Step-size for the optimization step;
    print("Step Size for the optimization : {}".format(step_size))
    

    # Value of sparsity or selection criterion
    Kmax = args.kmax
    print(f'K-max selection: {Kmax}')

    # Total Accuracies
    query_accuracies = []

    # Query_losses
    query_losses = []
    
    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()
    global_acc_hard = 101
    global_weights = None 
    global_class_wise_weights = None 

    # Global easy accuracy for easy episodes
    global_acc_easy = -1

    # 
    # Total number of epochs
    for epoch in range(0, n_epochs):
        # Compute the loss for the images with respect to the model 
        fp16 = True 
        
        # Create the class-wise-weights (1,5,1)
        #class_weights = [0,0,0,0,0]

        # Torch tensor for classes
        class_weights = torch.zeros((1,ways), device=args.device)
        # Denominator for each class
        for i in range(0, len(model.weights[0])):
            class_weights[0][support_labels[0][i]] += model.weights[0][i]#.item()
        

        # Reshape the torch tensor
        class_weights = torch.reshape(class_weights, (1,ways,1))
        # Summation of the total weights for the given class
        model.class_wise_weights = class_weights

        # Output of the model ---- #
        output = model.forward_light(support_images, support_labels, query_images) # Changed

        # Output 
        output = output.view(query_images.shape[0] * query_images.shape[1], -1)
        
        # Compute the loss 
        query_labels = query_labels.view(-1)
        loss = criterion(output, query_labels)
        loss_value = loss.item()

        # Zero out any accumulated gradients 
        model.zero_grad()
        loss.backward()

        # Update the weights -- Gradient ascent to maximise the loss
        weight_gradients = model.weights.grad # w.grad


        """ ####################  GRADIENT ASCENT STEP  #################### """
        if args.easy == True:
            print("########## EASY SET EXTRACTION ############ ")
            curr_weights = model.weights - step_size*100 * weight_gradients
        
        else:
            print("########## DIFFICULT SET EXTRACTION #############")
            # Weights after gradient ascent -- Tuning the step-size is important
            curr_weights = model.weights + step_size*100 * weight_gradients
        
        """ ##################### PROJECTION STEP ####################  """
        curr_weights_np = curr_weights.detach().cpu().numpy()

        # If joint-optimization is False -- then 
        if args.joint_opt == False:
            # projected weights
            curr_weights_np  = projection_step(args, curr_weights_np, Kmax)

        else:
            # Joint optimization
            labels_ = support_labels[0].detach().cpu().numpy()
            total_weights = [0]*len(labels_)
            # Unique labels for iteration
            unique_labels = np.unique(labels_)

            for label_ in unique_labels:
                # Positions where the label_ is hit
                pos = np.where(labels_ == label_)[0]
            
                # Extract the weights accordingly
                temp_weight = curr_weights_np[0][pos]

                # Projected weights for each of the classes to sparsify it 
                projected_weights = projection_step(args, temp_weight.reshape(1,-1), Kmax)
                
                c = 0
                for p_ in pos:
                    total_weights[p_] = projected_weights[c] 
                    c+= 1
            

            # Current weights
            curr_weights_np = np.array(total_weights)


        # Check if clamp parameter is True / False 
        if args.clamp == True:
            #print("Clamping.....")
            # Only keep the positive weights
            curr_weights_np[curr_weights_np <=0] = 0
        
        # Convert to tensor
        updated_weights = torch.tensor(curr_weights_np.reshape(1,-1), requires_grad=True, device=args.device)

        # Updating the model.weights with the weights obtained by gradient ascent and projection step
        model.weights = updated_weights
        
        # Accuracy before the projection steps from FastDiffSel
        acc1, acc5 = accuracy(output, query_labels, topk=(1, 5))

        # 
        if epoch %150 == 0:
            step_size = step_size * 0.5

        
        # Epoch
        if epoch % 50 == 0:
           # Class-wise-acc
            cl_wise_acc, total = class_wise_acc(output, query_labels, ways)

            old_weights = model.weights
            old_class_wise_weights = model.class_wise_weights
            acc1_hard_episode, cl_wise_acc, _ = compute_hard_ep_acc(args, model, support_images, support_labels, query_images, query_labels, support_indexes_total, class_keys, class_h5_dict, base_source, k, num_support_to_sample, save = False)
            
            # Easy Task
            if args.easy == True:
                if acc1_hard_episode >= global_acc_easy:
                    global_acc_hard = acc1_hard_episode
                    #print(old_weights)
                    global_weights = old_weights 
                    global_class_wise_weights = old_class_wise_weights

            # Hard Task
            else:
                # Check for the hard episode extraction
                if acc1_hard_episode <= global_acc_hard:
                    #print("############")
                    #print(old_weights)
                    global_acc_hard = acc1_hard_episode
                    global_weights = old_weights 
                    global_class_wise_weights = old_class_wise_weights
            
            # Model weights
            model.weights = old_weights
            model.class_wise_weights = old_class_wise_weights

            #print(f'Hard ep acc: {acc1_hard_episode}')
            print(f'Epoch: {epoch}; Current Loss: {loss_value}; Acc: {acc1}, Class-wise Acc: {cl_wise_acc}; Extracted hard ep acc: {acc1_hard_episode}')

        
        # Append the accuracies to the query 
        query_accuracies.append(acc1.item())
        # Append the loss to the query
        query_losses.append(loss_value)


        break 

    
    # Model weights
    model.weights = global_weights # Global weights
    model.class_wise_weights = global_class_wise_weights # Class-wise weights
    acc1_hard_episode, cl_wise_acc , hard_indexes = compute_hard_ep_acc(args, model, support_images, support_labels, query_images, query_labels, support_indexes_total, class_keys, class_h5_dict, base_source, k, num_support_to_sample, save=False)
    

    # Extracting the support images and saving them
    support_images_to_save = [support_original_images[index] for index in hard_indexes] #support_original_images[hard_indexes]
    support_labels_to_save = [support_original_labels[index] for index in hard_indexes] #support_original_labels[hard_indexes]
    
    # Support Transformation to save
    support_transformed_to_save = [support_transformed[index] for index in hard_indexes]


    # Easiest episode
    if args.easy:
        print(f'Accuracy of the easiest extracted few-shot task: {acc1_hard_episode} and the class-wise accuracies are: {cl_wise_acc}')
    else:
        print(f'Accuracy of the difficult extracted few-shot task: {acc1_hard_episode} and the class-wise accuracies are: {cl_wise_acc}')


    # 

    return acc1_hard_episode, query_accuracies, query_losses, support_images_to_save, support_labels_to_save, support_transformed_to_save


# Class sampler for the base source
def run_algorithm(args, base_source, model, transform):
    
    # Data / Episodic config part
    data_config = config_lib.DataConfig(args) # Type: Class
    episod_config = config_lib.EpisodeDescriptionConfig(args) # Type: Class
    
    # 
    use_dag_ontology_list = [False]
    use_bilevel_ontology_list = [False]
    datasets = [base_source]

    # Fix number of ways only on one dataset so far 
    if base_source == 'ilsvrc_2012':
        use_dag_ontology_list = [True]
    
    elif base_source == 'omniglot':
        use_bilevel_ontology_list = [True]
    

    ### Assign the episode description ###
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list
    
    # Dataset records path
    dataset_records_path = os.path.join(data_config.path, base_source)
    print("Meta-dataset record file: {}".format(dataset_records_path))
    dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

    # Use the test split
    split = Split.TEST

    # Number of classes
    num_classes = len(dataset_spec.get_classes(split=split))

    # The total number of classes
    print(f"=> There are {num_classes} classes in the {split} split of the combined datasets")
    
    # base path
    base_path = dataset_spec.path

    # Class set
    class_set = dataset_spec.get_classes(split) # class ids in this split #### Type: Class
    num_classes = len(class_set)

    # Base Path
    print("Base path: {}".format(base_path))
    # Class Set
    print("Class set: {}".format(type(class_set)))

    # Record File Pattern 
    record_file_pattern = dataset_spec.file_pattern
    
    # Checking if the record_file_pattern is correct or not
    assert record_file_pattern.startswith('{}'), f'Unsupported {record_file_pattern}.'
    
    # Dataset name
    dataset_name = base_source

    #print(f'Dataset name: {dataset_name}')
    # Initialise the class map
    class_map = {}

    # Initialise the class h5 dict
    class_h5_dict = {}

    # Initialise the class images
    class_images = {}
    class_samplers = {}

    # Class map
    class_map[dataset_name] = {}
    class_h5_dict[dataset_name] = {}
    class_images[dataset_name] = {}

    # The loop stores the path of each class of the h5 file
    for class_id in class_set:
        data_path = os.path.join(base_path, record_file_pattern.format(class_id))
        class_map[dataset_name][class_id] = data_path.replace('tfrecords', 'h5') # Replace tf-records with h5 file
        class_h5_dict[dataset_name][class_id] = None # closed h5 is None
        class_images[dataset_name][class_id] = [str(j) for j in range(dataset_spec.get_total_images_per_class(class_id))] # Stores the images / class

    # Class Samplers : Type -> Class
    class_samplers[dataset_name] = sampling.EpisodeDescriptionSampler(
                dataset_spec=dataset_spec,
                split=split,
                episode_descr_config=episod_config,
                use_dag_hierarchy=episod_config.use_dag_ontology_list[0],
                use_bilevel_hierarchy=episod_config.use_bilevel_ontology_list[0],
                ignore_hierarchy_probability=args.ignore_hierarchy_probability)
    

    
    # Sampler for the dataset
    sampler = class_samplers[dataset_name]
    num_ways = 5 # Fixed

    # Accuracies of hard-episodes
    hard_episodes_acc = []
    q_accs = []
    q_losses = []


    # Iterate over the number of tasks which need to be sampled
    for k in range(0,args.query_opt):
        # Episode description - sampler from MD's original sampling protocol
        episode_description = sampler.sample_episode_description()
        
        # Here make changes if you want to make number of support examples constant per class
        episode_description = tuple( # relative ids --> abs ids -- important
            (class_id + sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description)
        

        # Original Episode description
        #print("Original episode description: {}".format(episode_description))

        # Episode description
        #episode_description = episode_description[:num_ways]
        #print("Description of episode : {}".format(episode_description))

        # Episode classes - Class Ids for the current sample  
        episode_classes = list({class_ for class_, _, _ in episode_description})
        
        # Query images / labels
        query_images = []
        query_labels = []
        
        # Task
        print(f'Task : {k+1} with {len(episode_classes)} classes')

        # Support Pool
        support_pool = {}

        # Query Pool
        query_set = {}
        for class_id, _, _ in episode_description:
            support_pool[class_id] = None
            query_set[class_id] = []
        
        # Save path
        #save_path = '/home/t-sambasu/intern/PMF/metadataset_pmf/optimization_vis/' + base_source
        # print(save_path)

        # Query Indexes
        query_indexes = []
        query_label_indexes = []


        # Query Original Images
        query_original_images = []
        # Query Original Labels
        query_original_labels = []

        # Concatenate the number of supports
        total_num_supports = []


        # Iterate through the episode_description to read the support images and the query images
        for class_id, nb_support, nb_query in episode_description:
            #print(nb_support)
            assert nb_support + nb_query <= len(class_images[base_source][class_id]), \
                f'Failed fetching {nb_support + nb_query} images from {base_source} at class {class_id}.'
            
            # This randomly shuffles the image id's corresponding to the class_id
            random.shuffle(class_images[base_source][class_id])
            
            # For MS-COCO add a cap for the number of support examples to extract
            if base_source == 'quickdraw' or base_source == 'mscoco':
                num_classes = len(episode_classes)

                if num_classes <=20:
                    support_pool[class_id] = class_images[base_source][class_id][nb_query:800]
                
                else:
                    support_pool[class_id] = class_images[base_source][class_id][nb_query:100]
        

            # No cap for datasets other than MSCOCO or Quickdraw
            else:
                # Support Pool for the class which will be used to extract the hard support sets
                support_pool[class_id] = class_images[base_source][class_id][nb_query:] 



            # Extract the first few indexes as the query examples
            for j in range(0, nb_query):
                # Get the h5 files for the corresponding class from the base-source 
                h5_path = class_map[base_source][class_id] 
                
                # h5 dictionary for the class
                if class_h5_dict[base_source][class_id] is None: # will be closed in the end of main.py
                    class_h5_dict[base_source][class_id] = h5py.File(h5_path, 'r') # Extract the h5 file for each class

                # h5 file - For each class - extract
                h5_file = class_h5_dict[base_source][class_id]
                record = h5_file[class_images[base_source][class_id][j]] # Get the corresponding record with the id 
                # Note for accessing the h5 files, 
                x = record['image'][()]

                # Read the image 
                x = Image.fromarray(x)

                # Transform (check for this) - till now not required
                query_images.append(transform(x)) # Applying the transform to each of the query image
                query_labels.append(class_id)

                query_set[class_id].append(class_images[base_source][class_id][j])

                # Append the original images
                query_original_images.append(x)
                # Append the class_id
                query_original_labels.append(class_id)


                
            # Store the marker for (class_id) and (number_of_support)
            total_num_supports.append((class_id, nb_support))

        
        # Total Number of Supports
        print(f'Total Number of Supports : {total_num_supports}')

        # # Extracting the worst case support sets
        print("######## Extracting the worst case support sets via FastDiffSel #######")

         # Change the model parameters to False
        for name, param in model.named_parameters():
            param.requires_grad = False
        

        # Call the algorithm to generate hard support sets: 
        #acc1_hard_episode, query_accuracies, query_losses, support_images_to_save, support_labels_to_save, support_transformed_to_save = generate_hard_support_sets_optimization(args, query_images, query_labels, episode_description, class_map, class_h5_dict, support_pool, model, transform, k, total_num_supports)
        acc1_hard_episode, query_accuracies, query_losses, support_images_to_save, support_labels_to_save, support_transformed_to_save = generate_hard_support_sets_optimization(args, query_images, query_labels, episode_description, class_map, class_h5_dict, support_pool, model, transform, k, total_num_supports)

        #### Save the few-shot task ####
        task = [support_images_to_save, support_labels_to_save, query_original_images, query_original_labels] 
        
        # TODO: Save the task in the appropriate directory
        




# Main function
def main(args):
    device = torch.device(args.device)
    print("Device: {}".format(device))
    
    # fix the seed for reproducibility
    seed = args.seed
    print("Seed: {}".format(seed))
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data-config
    data_config = config_lib.DataConfig(args)

    # Transform for Test --- Define
    transform = test_transform(data_config)
    
    #### Define the model #### 
    model = get_model(args) # By default - DINO model (DINO + meta-training on MD ilsvrc-2012 split can also be used)
    model.to(device)

    # Cudnn benchmark
    cudnn.benchmark = True
    print("Dataset Used : {}".format(args.dataset))
    
    # Base source to sample the episodes from 
    base_source = args.base_sources[0]
    print("Base source for extracting: {}".format(base_source))

    # Get the dataset info --- generate the worst case support sets
    run_algorithm(args, base_source, model, transform)

    return 


# Main function
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print("Arguments: {}".format(args))
    main(args)