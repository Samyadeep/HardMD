""" 
Weighted prototypes for selecting the relevant support examples

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNetWeighted(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.backbone = backbone
        
        # Weights for each of the support examples in the class prototypes
        self.weights = None 

        # Class-wise weight vector
        self.class_wise_weights = None 


    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """

        # Number of classes
        num_classes = supp_y.max() + 1 # NOTE: assume B==1

        # B, nSupp, C, H, W
        B, nSupp, C, H, W = supp_x.shape

        # Multiply with the weights for the prototypes 
        with torch.no_grad(): # Detach gradients from the support to save memory
            supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))


        # Reshape the embeddings
        supp_f = supp_f.view(B, nSupp, -1) # [1, num_support, dimension]; weights dimension: [1, num_support]
        
        # Broadcasting operation here --- the weight vector has gradient enabled --- Multiplying the embeddings with the weight vector
        if self.weights != None: 
            # Multiply with the weights
            supp_f = supp_f * self.weights[..., None]  # supp_f shape: [B, nSupp, 384]; self.weights: [B ,num_examples, 1]
        
        # else:
        #     print("Protonets without weights")
            
        # One-hot encoding 
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # [B, nC, nSupp]
        #print(supp_y_1hot.sum(dim=2, keepdim=True))
        
        # B, nC, nSupp x B, nSupp, d = [B, nC, d]
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # This need not be touched, as it's only summing up the embeddings for each class which are already weighted
        
        #print(supp_y_1hot.sum(dim=2, keepdim=True).shape) # (B, nC, 1)
        
        # 
        if self.class_wise_weights != None:
            prototypes = prototypes / self.class_wise_weights #supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
        else:
            #print("No weights....")
            prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)
        

        # Embeddings for the query
        with torch.no_grad(): # Detach gradients from the query examples
            feat = self.backbone.forward(x.view(-1, C, H, W))
        
        # Features
        feat = feat.view(B, x.shape[1], -1) # B, nQry, d

        # Logits wrt prototypes and query
        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC

        return logits
    

    # Modified Forward Pass -- Light-weight 
    def forward_light(self, supp_x, supp_y, x):
        #print("########### Into forward Light ##############")
        # Number of classes
        num_classes = supp_y.max() + 1 # NOTE: assume B==1

        # B, nSupp, E (Batch_size, Number of Support, Size of the Embedding)
        B, nSupp, E = supp_x.shape

        # Support_f
        supp_f = supp_x # Shape : (B, nSupp, embedding_dim)
        

        # Broadcasting operation here --- the weight vector has gradient enabled --- Multiplying the embeddings with the weight vector
        if self.weights != None: 
            # Multiply with the weights
            supp_f = supp_f * self.weights[..., None]  # supp_f shape

        
        # One-hot encoding 
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # [B, nC, nSupp]
        #print(supp_y_1hot.sum(dim=2, keepdim=True))
        
        # B, nC, nSupp x B, nSupp, d = [B, nC, d]
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # This ne


        # 
        if self.class_wise_weights != None:
            prototypes = prototypes / self.class_wise_weights #supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
        else:
            #print("No weights....")
            prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)
        

        # Feature of the query 
        feat = x # [B, n_q, d]

        # 
        # Logits wrt prototypes and query
        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC


        return logits
