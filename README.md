# DFEI
Tensorflow implementation of Automatic Domain Feature Extraction and Personalized Integration (DFEI) framework.

Code for the paper:
Large-Scale Multi-Domain Recommendation: an Automatic Domain Feature Extraction and Personalized Integration Framework.

# Requirement
python==3.6

tensorflow-gpu==1.15.0

# Example to run the codes
```
python DFEI.py  --domains [0,1,2,4,5,6] --use_domainid 0 --verbose 200 --batch_size 512 --file_folder ./data --embedding_dim 16 --layers [128,64,32] --keep_prob [0.9,0.8,0.7] --lr 1e-3 --decay 0.9
```

The instruction of commands has been clearly stated in the codes (see the parse_args function).

# Dataset
We use the public KuaiRand-1K dataset [https://github.com/chongminggao/KuaiRand].

Last Update Date: Apr. 12, 2024 (UTC+8)
