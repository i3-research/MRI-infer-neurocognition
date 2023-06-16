import argparse
from datetime import datetime
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import models

from .dataset import AgePredictionDataset
from .vgg import VGG8

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SPLITS_DIR = os.path.join(ROOT_DIR, "folderlist2")
START_TIME = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
LOG_FILE = ''

def log(message):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, 'a+') as log_file:
        log_file.write(f"{message}\n")

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str, choices=['resnet18', 'vgg8'])
    parser.add_argument('--im_type', type=str, choices=['int', 'rav', 'int_rav'])
    parser.add_argument('--n_slices', type=int, default=130, help='number of slices')
    parser.add_argument('--iq', type=str, choices=['all', 'fiq', 'viq', 'piq'])
    parser.add_argument('--iq_type', type=str, choices=['absolute', 'residual'])
    parser.add_argument('--job-id', type=str, required=True, help='SLURM job ID')

    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--n-epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--initial-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')

    parser.add_argument('--sample', type=str, choices=['none', 'over', 'under', 'scale-up', 'scale-down'], default='none', help='sampling strategy')
    parser.add_argument('--reweight', type=str, choices=['none', 'inv', 'sqrt_inv'], default='none', help='reweighting strategy')
    parser.add_argument('--bin-width', type=int, default=1, help='width of age bins')
    parser.add_argument('--min-bin-count', type=int, default=10, help='minimum number of samples per age bin (bins with fewer samples will be removed)')
    parser.add_argument('--oversamp-limit', type=int, default=126, help='maximum number of samples each age bin will be inflated to when oversampling (bins with more samples will be cut off)')

    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

    ## For testing purposes only
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--eval', type=str, help='evaluate a pretrained model')
    parser.add_argument('--max-samples', type=int, help='limit the number of samples used for training/validation')
    parser.add_argument('--print-bin-counts', action='store_true', help='print age bin counts and exit')

    args = parser.parse_args()
    assert (args.sample == 'none') or (args.reweight == 'none'), "--sample is incompatible with --reweight"

    global LOG_FILE
    LOG_FILE = os.path.join(ROOT_DIR, "logs", f"{args.job_id}.log")
    try:
        os.remove(LOG_FILE)
    except FileNotFoundError:
        pass

    for arg in vars(args):
        log(f"{arg}: {getattr(args, arg)}")
    log('='*20)
    return args

def load_samples(split_fnames, im_type):
    def correct_path(path):
        path = path.replace('/ABIDE/', '/ABIDE_I/')
        path = path.replace('/NIH-PD/', '/NIH_PD/')
        if im_type == "rav":
            path = path.replace('/reg_', '/ravens_')
        return path

    schema = {'id': str, 'fiq': float, 'fiq_r': float, 'piq': float, 'piq_r': float, 'viq': float, 'viq_r': float, 'sex': int, 'dg': int, 'age': float, 'path': str, 'site': int}
    dfs = []
    for fname in split_fnames:
        df = pd.read_csv(fname, sep=' ', header=None, names=['id', 'fiq', 'fiq_r', 'piq', 'piq_r', 'viq', 'viq_r', 'sex', 'dg', 'age', 'path', 'site'], dtype=schema)
        df['path'] = df['path'].apply(correct_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)

    return combined_df

def setup_model(arch, no_of_slices, n_classes, device):
    if arch == 'resnet18':
        model = models.resnet18(num_classes=n_classes)
        # Set the number of input channels to 130
        #model.conv1 = nn.Conv2d(130, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = nn.Conv2d(no_of_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif arch == 'vgg8':
        model = VGG8(in_channels=no_of_slices, num_classes=n_classes)
    else:
        raise Exception(f"Invalid arch: {arch}")
    model.double()
    model.to(device)
    return model

def train(model, optimizer, train_loader, device):
    def weighted_l1_loss(inputs, targets, weights):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss
    def l1_loss(inputs, targets):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss = torch.mean(loss)
        return loss

    model.train()

    losses = []

    for batch_idx, (images, iq, weights) in enumerate(train_loader):
        images, iq, weights = images.to(device), iq.to(device), weights.to(device)
        optimizer.zero_grad()
        iq_preds = model(images).view(-1)
        iq = torch.flatten(iq)  
        loss = l1_loss(iq_preds, iq)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 10 == 0:
            log(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
    
    return np.mean(losses)

def validate(model, val_loader, device):
    model.eval()

    losses = []
    all_preds = []

    with torch.no_grad():
        for (images, ages, _) in val_loader:
            # When batch_size=1 DataLoader doesn't convert the data to Tensors
            if not torch.is_tensor(images):
                images = torch.tensor(images).unsqueeze(0)
            if not torch.is_tensor(ages):
                ages = torch.tensor(ages).unsqueeze(0)
            images, ages = images.to(device), ages.to(device)
            age_preds = model(images).view(-1)
            loss = F.l1_loss(age_preds, ages, reduction='mean')

            losses.append(loss.item())
            all_preds.extend(age_preds)
    
    return np.mean(losses), torch.stack(all_preds).cpu().numpy()

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    opts = parse_options()

    log(f"Starting at {START_TIME}")

    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints", opts.eval or opts.job_id)
    results_dir = os.path.join(ROOT_DIR, "results", opts.job_id)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if opts.iq == 'all':
        n_classes = 3
    else:
        n_classes = 1

    log("Setting up dataset")
    
    data_list = glob.glob(f"{SPLITS_DIR}/all_iq_data_with_residuals_allok.list")
    df = load_samples(data_list, opts.im_type)
    kf5 = KFold(n_splits = 5, random_state = 1, shuffle = True)
    
    i = 1
    for fold in kf5.split(df):
        training = df.iloc[fold[0]]
        validation = df.iloc[fold[1]]

        train_dataset = AgePredictionDataset(training, iq=opts.iq, iq_type=opts.iq_type, n_slices=opts.n_slices, im_type=opts.im_type, reweight=opts.reweight, lds=opts.lds, lds_kernel=opts.lds_kernel, lds_ks=opts.lds_ks, lds_sigma=opts.lds_sigma)
        val_dataset = AgePredictionDataset(validation, iq=opts.iq, iq_type=opts.iq_type, n_slices=opts.n_slices, im_type=opts.im_type)

        train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1)

        ## Dump configuration info to a JSON file

        with open(f"{results_dir}/config.json", 'w+') as f:
            cfg = vars(opts)
            cfg['start_time'] = START_TIME
            json.dump(cfg, f, sort_keys=True, indent=4)

        log("Setting up model")

        device = torch.device('cpu' if opts.cpu else 'cuda')
        model = setup_model(opts.arch, opts.n_slices, n_classes, device)
        optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr, weight_decay=opts.weight_decay)
        scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

        ## Training and evaluation
        if opts.arch == 'resnet18':
            prefix = '2drn'
        else:
            prefix = '2dvgg'

        if opts.eval is None:
            log("Starting training process")

            best_epoch = None
            best_val_loss = np.inf
            train_losses = []
            val_losses = []

            for epoch in range(opts.n_epochs):
                log(f"Epoch {epoch}/{opts.n_epochs}")
                train_loss = train(model, optimizer, train_loader, device)
                log(f"Mean training loss: {train_loss}")
                val_loss, _ = validate(model, val_loader, device)
                log(f"Mean validation loss: {val_loss}")
                scheduler.step()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{checkpoint_dir}/{opts.im_type}_{prefix}_{opts.iq}_{opts.n_slices}sl_{opts.iq_type}_f{i}.pth")
                    best_epoch = epoch
                    best_val_loss = val_loss

            log(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")

        log("Evaluating best model on val dataset")

        checkpoint = torch.load(f"{checkpoint_dir}/{opts.im_type}_{prefix}_{opts.iq}_{opts.n_slices}sl_{opts.iq_type}_f{i}.pth", map_location=device)
        model.load_state_dict(checkpoint)

        _, best_val_preds = validate(model, val_dataset, device)

        ## Save results so we can plot them later

        log("Saving results")
 
        if opts.eval is None:
            np.savetxt(f"{results_dir}/train_losses_over_time_f{i}.txt", train_losses)
            np.savetxt(f"{results_dir}/val_losses_over_time_f{i}.txt", val_losses)

        val_df_with_preds = validation.copy()
        if opts.iq == 'all':
            id_fiq = list(range(0,510,3))
            id_viq = list(range(1,510,3))
            id_piq = list(range(2,510,3))
            val_df_with_preds['fiq_pred'] = best_val_preds[id_fiq]
            val_df_with_preds['piq_pred'] = best_val_preds[id_piq]
            val_df_with_preds['viq_pred'] = best_val_preds[id_viq]
        else:
            val_df_with_preds[f"{opts.iq}_pred"] = best_val_preds
            
        val_df_with_preds.to_csv(f"{results_dir}/{opts.im_type}_{prefix}_{opts.iq}_{opts.n_slices}sl_{opts.iq_type}_f{i}.csv", index=False)
        i = i + 1

if __name__ == '__main__':
    main()
