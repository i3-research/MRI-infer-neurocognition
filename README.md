# Structural MRI Inferring Human Intelligence

This is the code repository for our [paper](#cite) titled "[Can deep learning predict human intelligence from structural brain MRI?](https://www.biorxiv.org/content/10.1101/2023.02.24.529924v1)" that  tests whether deep learning of sMRI can predict an individual subject’s verbal, comprehensive, and full-scale intelligence quotients (VIQ, PIQ, FSIQ), which reflect both fluid and crystallized intelligence.

[PDF Link](https://www.biorxiv.org/content/10.1101/2023.02.24.529924v1.full.pdf) | [DOI](https://doi.org/10.1101/2023.02.24.529924)

## Repository layout
```
|- 2d_cnn    # Source codes of 2D CNN-based IQ prediction approach
|- 3d_cnn    # Source codes of 2D CNN-based IQ prediction approach (requires MONAI [https://github.com/Project-MONAI/MONAI] installation) 
```

## Motivation

- T1-weighted structural brain magnetic resonance images (sMRI) have been correlated with intelligence. 
- Nevertheless, population-level association does not fully account for individual variability in intelligence. 
- To address this, individual prediction studies emerge recently. However, they are mostly on predicting fluid intelligence (the ability to solve new problems). 
- Studies are lacking to predict crystallized intelligence (the ability to accumulate knowledge) or general intelligence (fluid and crystallized intelligence combined). 

## Brief Description of the Method
We trained two 2D and four 3D deep CNNs using T1-weighted MRI volumes (N = 850) in two settings. In the first setting, we used intensity (i.e., contrast channel) and RAVENS (regional analysis of volumes examined in normalized space; i.e., morphometry channel) images to predict three IQ scores separately (i.e., FSIQ or PIQ, or VIQ). On the other hand, we used intensity and RAVENS maps to predict three IQ scores simultaneously (i.e., FSIQ and PIQ, and VIQ) in the second setting. For 2D CNNs, we chose a different number of axial slices (n = [5, 10, 20, 40, 70, 100, 130]) as channels.

## Table of contents
1. [Installation](#installation)
2. [Usage](#usage)
4. [Cite](#cite)


<a name="installation"></a>
### Installation
We used [MONAI](https://github.com/Project-MONAI/MONAI) framework for our 3D CNN-based experiments. To run ```3d_cnn\3d_cnns_GradCAM.ipynb.py``` and ```3d_cnn\iq_prediction_3d_cnns.py```, MONAI installation is required. Detailed instructions on how to install MONAI can be found [here](https://docs.monai.io/en/latest/installation.html).  


<a name="usage"></a>
### Usage
#### Data List Preparation
In our experiements, we generated a list for NIFTI data and associated ground truth scores as below:
```
[patient_id absolute_FIQ residual_FIQ absolute_PIQ residual_PIQ absolute_VIQ residual_VIQ sex diagnosis age path site_id]
```
For example:
```
51493 102 -13 103 -8 101 -13 2 2 29.2 /neuro/labs/grantlab/research/MRI_Predict_Age/ABIDE/2NIFTI_SS_SEG_RAVENS/ABIDE51493/reg_ABIDE51493_MPRAGE_ss_to_SRIatlas.nii.gz 1
50642 103 -7 107 -3 98 -9 1 1 33.0 /neuro/labs/grantlab/research/MRI_Predict_Age/ABIDE/2NIFTI_SS_SEG_RAVENS/ABIDE50642/reg_ABIDE50642_MPRAGE_ss_to_SRIatlas.nii.gz 2
...
...
```

#### 2D CNNs
To train 2D CNNs (i.e., ResNet18 and VGG8) in a 5-fold cross-validation setup, run ```2d_cnn\iq_prediction_2d_cnn.py``` as:
```
python -m iq_prediction_2d_cnn.py <architecture> --im_type <input type> --n_slices <no of slices> --iq <IQ to be predicted> -- iq_type <absolute or residual>
```
Options are available in ```2d_cnn\iq_prediction_2d_cnn.py``` as:
```
parser.add_argument('arch', type=str, choices=['resnet18', 'vgg8'])
parser.add_argument('--im_type', type=str, choices=['int', 'rav', 'int_rav'])
parser.add_argument('--n_slices', type=int, default=130, help='number of slices')
parser.add_argument('--iq', type=str, choices=['all', 'fiq', 'viq', 'piq'])
parser.add_argument('--iq_type', type=str, choices=['absolute', 'residual'])
```

To generate GradCAM images, use ```2d_cnn\iq_inference_resnet18_vgg8_gradcam.ipynb```.


#### 3D CNNs
To train 3D CNNs (i.e., ResNet18, ResNet50, DenseNet121, and DenseNet169) in a 5-fold cross-validation setup, run ```3d_cnn\iq_prediction_3d_cnn.py``` as:
```
python iq_prediction_3d_cnn.py --arch <architecture> --im_type <input type> --iq <IQ to be predicted> -- iq_type <absolute or residual> --folds <fold number>
```
Options are available in ```3d_cnn\iq_prediction_3d_cnn.py``` as:
```
parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50', 'densenet121', 'densenet169'])
parser.add_argument('--im_type', type=str, choices=['int', 'rav', 'int_rav'])
parser.add_argument('--iq', type=str, choices=['all', 'fiq', 'viq', 'piq'])
parser.add_argument('--iq_type', type=str, choices=['absolute', 'residual'])
parser.add_argument('--folds', type=int, default=0, help='fold number')
```

To generate GradCAM images, use ```3d_cnn\3d_cnns_GradCAM.ipynb```.

<a name="cite"></a>
### Cite
```bibtext
@article{hussain2023can,
  title={Can deep learning predict human intelligence from structural brain MRI?},
  author={Hussain, Mohammad Arafat and LaMay, Danielle and Grant, Ellen and Ou, Yangming},
  journal={bioRxiv},
  pages={2023--02},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
