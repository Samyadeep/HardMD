""" Implementation of ViTs with PEFT methods

Pros: 
- Dynamic adapters and compacters which can be inserted amongst any of the transformer block
- Our contribution: We learn a task-specific position of the adapters, which results in less number of adapters to be inserted across the network
- Results on MetaDataset, VTAB, ORBIT and CDFSL benchmarks

"""

# Libraries
import torch
import torch.nn as nn

import math
from functools import partial
from .utils import trunc_normal_
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 
        
        # Scale
        self.scale = qk_scale or head_dim ** -0.5
        
        # Linear Model for computing the attention vectors
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # 
    def forward(self, x):
        # Shape of the input: (B, Number of Patches (including [CLS] token, Embedding dimension))
        B, N, C = x.shape

        # Main module : 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Get the query, key and value embeddings
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention vector
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Attention weights
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weight the value with the attention weights
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Projection layer
        x = self.proj(x)

        # Whether to apply dropout or not
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x, return_attention=False):
        # Self-attention module
        y, attn = self.attn(self.norm1(x))

        # Only if attention needs to be returned
        if return_attention:
            return attn
        
        # Add the first-residual link
        x = x + self.drop_path(y)

        # Add the second-residual link
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



############################# Transformer Block with Adapter ####################################
""" Implementation of Transformer block with Adapter 
Extra parameters:
- adapter_dimension: Bottleneck downprojection layer
- layer_num: Layer number for the adapter
- top_layer_adapter: Indicator variable which enables injection of adapters only in the final layers after top_layer_adapter

Currently only one adapter block is inserted per block

"""
def get_nonlin_func(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")



class BottleneckAdapter(nn.Module):
    def __init__(self, adapter_dim = 32, dim=384):
        super().__init__()
        self.adapter_hidden_size = adapter_dim 
        self.adapter_input_size = dim 


        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_hidden_size)
        self.up_proj = nn.Linear(self.adapter_hidden_size, self.adapter_input_size)
        self.norm_ad = nn.LayerNorm(dim)

        self.init_weights()
    
    def init_weights(self):
        """ Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function """
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()


    def forward(self, x):
        #output = self.up_proj(F.gelu(self.down_proj(x)))
        output = self.up_proj(F.relu(self.down_proj(self.norm_ad(x))))
        output = x + output 
        return output



class BlockWithAdapter(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., adapter_dimension= 32, layer_num = 0, top_layer_adapter=-1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        ###### Configuration for the adapter block ######
        self.layer_num = layer_num 
        self.top_layer_adapter = top_layer_adapter
        self.adapter_dimension = adapter_dimension

        # Condition for adapter layer to kick in
        if self.layer_num >= self.top_layer_adapter:
            self.adapter_layer = BottleneckAdapter(adapter_dimension, dim)
            # self.adapter_norm = norm_layer(dim)
            #self.adapter_layer_inter = BottleneckAdapter(adapter_dimension, dim)
    # 
    def forward(self, x, return_attention=False):
        # Self-attention module
        y, attn = self.attn(self.norm1(x))

        # Only if attention needs to be returned
        if return_attention:
            return attn
        
        # Insert adapter after the multihead attention
        #y = self.adapter_layer_inter(y)

        # Add the first-residual link
        x = x + self.drop_path(y) # Residual link + multi-head attention

        # Arch till now: NORM -> ATTENTION -
        # Kick in the adapter layers only when 
        if self.layer_num >= self.top_layer_adapter:
            # print("Adaptation ##")
            x2 = self.mlp(self.norm2(x))

            # Adapter output
            adapter_output = self.adapter_layer(x2)

            # Adding the residual link
            x = x + self.drop_path(adapter_output)
            
            return x 
        
        # Add the second-residual link
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


##########################################################################################


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Embedding for the patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Output Shape from self.proj(x) : (1, 384, 14, 14) --> For each of the position, the dimension is 384
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        print("Drop rate: {}".format(drop_rate))
        print("Drop path rate: {}".format(drop_path_rate))
        print("Attention drop rate: {}".format(attn_drop_rate))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    # Preparation of tokens
    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)
        
        # By default drop_rate in pos_drop is zero.
        return self.pos_drop(x)
    
    
    # Forward function
    def forward(self, x, ada_token=None, use_patches=False):
        # Prepare tokens
        x = self.prepare_tokens(x, ada_token)
        
        # Iterate through all the blocks
        for blk in self.blocks:
            x = blk(x) # Inject adapters in this block layer
        
        # Layer norm after the final layer
        x = self.norm(x)

        if use_patches:
            # Embeddings of all the patches except for [CLS] token
            return x[:, 1:]
        else:
            # Embeddings of only the [CLS] token
            return x[:, 0]

    # Get self-attention from the last layer ---- could be modified to get attention from the other layers too
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    
    # Get intermediate layers 
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

################################################ Vision Transformer with Adapter ##################################################

""" Vision Transformer with Adapter Layer """
class VisionTransformerAdapter(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, adapter_layers = list(range(0,12)), adapter_dim = 32, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        # Layers where the adapter module can be injected
        self.adapter_layers = adapter_layers
        self.adapter_dimension = adapter_dim
        #print("Adapter layer: {}".format(self.adapter_layers))
        #print("Adapter Dimension: {}".format(self.adapter_dimension))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # This will be changed
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        #     for i in range(depth)])

        # Blocks for the transformer layer -- Default : Applies adapter in each block
        self.blocks = nn.ModuleList([
            BlockWithAdapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], adapter_dimension= adapter_dim, layer_num = i, top_layer_adapter= -1, norm_layer=norm_layer)
            for i in range(depth)])
        
        # Layer-norm
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    # Initialise the weights
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    # Preparation of tokens
    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)
        
        # By default drop_rate in pos_drop is zero.
        return self.pos_drop(x)
    
    
    # Forward function
    def forward(self, x, ada_token=None, use_patches=False):
        # Prepare tokens
        x = self.prepare_tokens(x, ada_token)
        
        # Iterate through all the blocks
        for blk in self.blocks:
            x = blk(x) # Inject adapters in this block layer
        
        # Layer norm after the final layer
        x = self.norm(x)

        if use_patches:
            # Embeddings of all the patches except for [CLS] token
            return x[:, 1:]
        else:
            # Embeddings of only the [CLS] token
            return x[:, 0]

    # Get self-attention from the last layer ---- could be modified to get attention from the other layers too
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    
    # Get intermediate layers 
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


################################################ Vision Transformer with Adapter ##################################################

# ViT Tiny model
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# Small ViT million parameters
def vit_small(patch_size=16, adapter=False, layers = None, adapter_dim = 32,  **kwargs):
    # Invoke the normal vision transformers
    if adapter == False:
        print("############### Training without adapter layers ####################")
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    # Invoke the vision transformer with adapter layer
    else:    
        print("############ Loading vision transformer (Small) with adapters ################")
        #print("Adapter Dim: {}".format(adapter_dim))
        model = VisionTransformerAdapter(
            patch_size=patch_size, adapter_layers = layers, adapter_dim = adapter_dim, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


    return model


# Base ViT ~ 86 million parameters
def vit_base(patch_size=16, adapter = False, layers=None, adapter_dim =32, **kwargs):
    if adapter == False:
        print("############### Training without adapter layers ####################")
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
    else:
        print("########## Loading Vision Transformers (Base) with Adapters #################")
        model = VisionTransformerAdapter(
            patch_size=patch_size, adapter_layers = layers, adapter_dim = adapter_dim, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    
    return model
