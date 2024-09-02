import os
import nibabel as nib
import pandas as pd
import numpy as np

"""
Modified from code from Gabriel Sarch
"""
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"] + "/result"

def save_1d_noise_ceiling(
    subjs,
    beta_path,
    save_dir="output/noise_ceiling",
):

    # from https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/h_T_2Djeid
    # How many distinct images were shown all three times to each subject?
    shown_three = {
        1: 10000,
        2: 10000,
        3: 6234,
        4: 5445,
        5: 10000,
        6: 6234,
        7: 10000,
        8: 5445,
    }
    # How many distinct images were shown at least twice to each subject?
    shown_two = {
        1: 10000,
        2: 10000,
        3: 8355,
        4: 7846,
        5: 10000,
        6: 8355,
        7: 10000,
        8: 7846,
    }
    # How many distinct images were shown at least once to each subject?
    shown_one = {
        1: 10000,
        2: 10000,
        3: 9411,
        4: 9209,
        5: 10000,
        6: 9411,
        7: 10000,
        8: 9209,
    }

    for subj in subjs:

        print("processing subject", subj)
        betas_subj_dir = "%s/subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR" % (
            beta_path,
            subj,
        )

        nc_file = nib.load(
            os.path.join(betas_subj_dir, "ncsnr.nii.gz")
        )  # this is noise ceiling signal-to-noise ratio (ncsnr) for each voxel and NOT noise ceiling
        ncsnr = nc_file.get_data()

        root = f"output/voxels_masks/subj{subj}/"

        cm = np.load(root + f"cortical_mask_subj0{subj}.npy")

        ncsnr_1d = ncsnr[cm]

        A = shown_three[subj]  # only three trials
        B = shown_two[subj] - A  # only two trials
        C = shown_one[subj] - (B + A)  # only one trials
        noise_ceiling_1d = 100 * (
            (ncsnr_1d) ** 2
            / ((ncsnr_1d) ** 2 + ((A / 3 + B / 2 + C / 1) / (A + B + C)))
        )

        print("Saving...")

        subj_save_root = "%s/subj%01d" % (save_dir, subj)

        if not os.path.exists(subj_save_root):
            os.mkdir(subj_save_root)

        # save it
        np.save(
            "%s/noise_ceiling_1d_subj%02d.npy" % (subj_save_root, subj),
            noise_ceiling_1d,
        )


if __name__ == "__main__":
    subjs = [1, 2, 3, 4, 5, 6, 7, 8]
    
    beta_path = config["NSD"]["BetaPath"]

    save_dir = ROOT + "/output/noise_ceiling"
    if not os.exist(save_dir):
        os.makedir(save_dir)
    save_1d_noise_ceiling(subjs=subjs, beta_path=beta_path, save_dir=save_dir)
