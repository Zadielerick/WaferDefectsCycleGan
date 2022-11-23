import numpy as np
import argparse
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import csv
import shutil

def load_images(image_dir, tensor_list):
    # Load PNG images into Tensors
    for child in Path.iterdir(Path(image_dir)):
        fileID = str(child.resolve())
        if fileID.endswith(".png"):
            image = Image.open(fileID)
            image_rgb = image.convert("RGB")
            transform = transforms.Compose([
                transforms.PILToTensor()
            ])
            tensor_list.append(transform(image_rgb))

def parse_options():
    parser = argparse.ArgumentParser(
                        prog = 'FID-KID-Calc',
                        description = 'Utility to calculate FID-KID scores',
                        epilog = 'Version 1.0')
    parser.add_argument('-t', '--test_dir', required=True, type=Path)

    opts = parser.parse_args()

    # Check input and output directory exist
    if (opts.test_dir.is_dir()):
        print('Calculating FID and KID scores for ' + str(opts.test_dir))
    else:
        print('Test directory does not exist')
        quit()
    
    return opts.test_dir

def get_fid_kid(real_dir, fake_dir):
#def get_fid_kid():
    #r_dir, f_dir = parse_options()
    r_dir = real_dir
    f_dir = fake_dir

    fid = FrechetInceptionDistance(feature=2048)

    r_tensors = []
    f_tensors = []

    load_images(r_dir, r_tensors)
    load_images(f_dir, f_tensors)

    r_stack = torch.stack(r_tensors)
    f_stack = torch.stack(f_tensors)

    # Calculate FID
    fid.update(r_stack, real=True)
    fid.update(f_stack, real=False)
    fid_score = fid.compute()

    # Calculate KID
    kid = KernelInceptionDistance(subset_size=50)
    kid.update(r_stack, real=True)
    kid.update(f_stack, real=False)
    kid_score = kid.compute()

    return (fid_score.item(), kid_score[0].item(), kid_score[1].item())

def main():
    test_dir = parse_options()
    csv_fields = ['Test Defect', 'Epoch', 'FID', 'KID mean', 'KID std']
    csv_rows = []
    for child in Path.iterdir(Path(test_dir)):
        fileID_parent = child.resolve()
        fileID = fileID_parent / 'images'

        fileID_real = fileID / 'real_B'
        fileID_fake = fileID / 'fake_B'
        fileID_real.mkdir(exist_ok=True)
        fileID_fake.mkdir(exist_ok=True)

        # Copy real and fake defect images to directories
        real_list = list(fileID.glob('*real_B.png'))
        for real in real_list:
            shutil.copy(str(real), str(fileID_real))

        fake_list = list(fileID.glob('*fake_B.png'))
        for fake in fake_list:
            shutil.copy(str(fake), str(fileID_fake))

        # Calculate FID and KID values from these directories
        fid_score, kid_score_mean, kid_score_std = get_fid_kid(fileID_real, fileID_fake)
        test_epoch = str(fileID_parent).split('/')[-1]
        test_epoch = test_epoch.split('_')[-1]
        test_name = str(test_dir).split('/')[-1]

        csv_row = [test_name, test_epoch, str(fid_score), str(kid_score_mean), str(kid_score_std)]
        csv_rows.append(csv_row)
    filename = str(test_dir)  + "/fid_kid.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_fields)
        csvwriter.writerows(csv_rows)
if __name__=="__main__":
    main()
