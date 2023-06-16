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

#### 2D CNNs

We also shared the code for the key algorithm (that runs inside the training loop) in ```mymodule\alnt.py```. 
- This code can be plugged into any training routine, after modifying lines 13 and 40 to load someone's own deep model. 
- To run as RGS method, assign ```alpha = 1``` in line 127 
- To run as RGM method, assign ```alpha = 0``` in line 127
- To run as RGS&M method, assign ```alpha = 0.5``` in line 127
- To run as RGS&M+AL method, uncomment line 128
- To find the gradient with respect to the last layer weights (for RGM, RGS&M, RGS&M+AL), find and replace the name of the last layer of your own model in line 68.

<a name="cite"></a>
### Cite
```bibtext
@article{hussain2022active,
  title={Active deep learning from a noisy teacher for semi-supervised 3D image segmentation: Application to COVID-19 pneumonia infection in CT},
  author={Hussain, Mohammad Arafat and Mirikharaji, Zahra and Momeny, Mohammad and Marhamati, Mahmoud and Neshat, Ali Asghar and Garbi, Rafeef and Hamarneh, Ghassan},
  journal={Computerized Medical Imaging and Graphics},
  doi = {10.1016/j.compmedimag.2022.102127},
  pages={102127},
  year={2022},
  publisher={Elsevier}
}
```
