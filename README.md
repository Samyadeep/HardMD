# HardMD++: Towards Understanding Few-Shot Performance on Difficult Tasks (ICLR 2023) [https://openreview.net/pdf?id=wq0luyH3m4]
In this repository, we introduce FastDiffSel, an efficient algorithm to extract difficult few-shot tasks from Meta-Dataset and other large-scale vision datasets (e.g., ORBIT, CURE-OR, ObjectNet). 


If you find our project helpful, please consider cite our paper:
```
@inproceedings{
basu2023hardmetadataset,
title={Hard-Meta-Dataset++: Towards Understanding Few-Shot Performance on Difficult Tasks},
author={Samyadeep Basu and Megan Stanley and John F Bronskill and Soheil Feizi and Daniela Massiceti},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=wq0luyH3m4}
}


```

### To extract difficult few-shot tasks from Meta-Dataset Domains:

```

python fastdiffsel_md.py --dataset=meta_dataset --base_sources=aircraft --data-path /fs/cml-datasets/Meta-Dataset --arch dino_small_patch16 --fp16 --device cuda:0  --deploy weighted --weighted_step_size 0.2 --optimizer_epoch 1 --kmax 10 --query_opt 5 --joint_opt --sup 5  --md_sampling

```


### Coming Soon!

#### data-loaders for extracted tasks from Meta-Dataset using FastDiffSel

### Dependencies 

```
pip install requirements.txt
```

#### To install the .h5 files in Meta-Dataset, follow the procedure in https://github.com/hushell/pmf_cvpr22

#### Correspondence to sbasu12@umd.edu