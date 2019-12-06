# HOb2sRNN
Multi-source (Radar-Optical) and Multi-temporal Land Cover Mapping at object level leveraging hierarchical class relationships.

Tested with Tensorflow 1.15. 

The implementation is described in the section 3 (Method) of this **[paper](https://arxiv.org/abs/1911.08815)**.

## How to use?
### Training and inference time
Check the running bash file (`run.sh`). The `HOb2sRNN.py` python file is waiting for 14 parameters :

`python HOb2sRNN.py rad_train_file opt_train_file gt_train_file rad_valid_file opt_valid_file gt_valid_file rad_test_file opt_test_file gt_test_file split_number model_outpath rad_timestamps opt_timestamps hier_pretraining`

1. **rad_train_file**: The radar array for training samples in numpy file format (`.npy`). 

2. **opt_train_file**: The optical array for training samples in numpy file format (`.npy`).  

3. **gt_train_file**: The ground truth array for training samples in numpy file format (`.npy`).  

1. **rad_valid_file**: The radar array for validation samples in numpy file format (`.npy`). 

2. **opt_valid_file**: The optical array for validation samples in numpy file format (`.npy`).  

3. **gt_valid_file**: The ground truth array for validation samples in numpy file format (`.npy`).  

1. **rad_test_file**: The radar array for test samples in numpy file format (`.npy`). 

2. **opt_test_file**: The optical array for test samples in numpy file format (`.npy`).  

3. **gt_test_file**: The ground truth array for test samples in numpy file format (`.npy`).  

10. **split_number**: A number indicating on which split of your datastet the model is trained and used in the model outname. Put any number if you are using only one split of the dataset.

10. **model_outpath**: The path where models are saved.

10. **rad_timestamps**: The radar time series length. 

10. **opt_timestamps**: The optical time series length. 

11. **hier_pretraining**: A parameter with value 1 or 2. Choose 1 to used the hierarchical pretraining strategy described in the paper or 2 to make a simple classification.

