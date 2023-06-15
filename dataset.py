import nibabel
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from torch.utils import data

from .utils import get_lds_kernel_window

def prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma):
    if reweight == 'none':
        return None
    
    bin_counts = df['iqbin'].value_counts()
    # num_per_label[i] = the number of subjects in the age bin of the ith subject in the dataset
    if reweight == 'inv':
        num_per_label = [bin_counts[bin] for bin in df['iqbin']]
    elif reweight == 'sqrt_inv':
        num_per_label = [np.sqrt(bin_counts[bin]) for bin in df['iqbin']]
    
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_value = pd.Series(
            convolve1d(bin_counts.values, weights=lds_kernel_window, mode='constant'),
            index=bin_counts.index)
        num_per_label = [smoothed_value[bin] for bin in df['iqbin']]

    weights = [1. / x for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

class AgePredictionDataset(data.Dataset):
    def __init__(self, df, iq, iq_type, n_slices, im_type, reweight='none', lds=False, lds_kernel='gaussian', lds_ks=9, lds_sigma=1, labeled=True):
        self.df = df
        self.iq = iq
        self.iq_type = iq_type
        self.n_slices = n_slices # my edits
        self.im_type = im_type # my edits
        self.weights = prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma)
        self.labeled = labeled

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.im_type == "int_rav": # my edits
            image = nibabel.load(row['path']).get_fdata() # my edits
            row['path'] = row['path'].replace('/reg_', '/ravens_') # my edits
            ravens = nibabel.load(row['path']).get_fdata() # my edits
            image = image + ravens # my edits
        else:
            image = nibabel.load(row['path']).get_fdata()
            
        #image = image[54:184, 25:195, 12:132] # Crop out zeroes --> original
        '''
        # Fixing the plane issue
        st = 120 - (self.n_slices//2)
        ed = st + self.n_slices
        image = image[st:ed, 25:195, 12:132] # # my edits; center 120th slice
        '''
        print(image.shape)
        st = 75 - (self.n_slices//2) #new edit
        ed = st + self.n_slices #new edit
        print(self.n_slices, st, ed)
        image = image[50:190, 25:195, st:ed] #new edit
        print(image.shape)
        image = np.transpose(image, (2, 0, 1)) #new edit
        
        
        image /= np.percentile(image, 95) # Normalize intensity

        if self.labeled:
            if self.iq_type == 'absolute':
                if self.iq == 'all':
                    iq = np.array([row['fiq'], row['viq'], row['piq']]) # my edits
                elif self.iq == 'fiq':
                    iq = np.array([row['fiq']])
                elif self.iq == 'piq':
                    iq = np.array([row['piq']])
                elif self.iq == 'viq':
                    iq = np.array([row['viq']])  
            else:
                if self.iq == 'all':
                    iq = np.array([row['fiq_r'], row['viq_r'], row['piq_r']]) # my edits
                elif self.iq == 'fiq':
                    iq = np.array([row['fiq_r']])
                elif self.iq == 'piq':
                    iq = np.array([row['piq_r']])
                elif self.iq == 'viq':
                    iq = np.array([row['viq_r']]) 
            weight = self.weights[idx] if self.weights is not None else 1.
            return (image, iq, weight)
        else:
            return image

    def __len__(self):
        return len(self.df)
