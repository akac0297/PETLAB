#!/usr/bin/env python
# coding: utf-8

import pathlib
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform

template_b50 = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_010/IMAGES/WES_010_TIMEPOINT_1_MRI_DWI_B50.nii.gz")
new_b50=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/WES_007_TIMEPOINT_1_MRI_DWI_B50.nii.gz")


template_b50_to_new_rigid, tfm_template_to_new = linear_registration(
    new_b50,
    template_b50,
    shrink_factors= [10,5],
    smooth_sigmas= [2,1],
    sampling_rate= 1,
    final_interp= 2,
    metric= 'mean_squares',
    optimiser= 'gradient_descent_line_search',
    reg_method='rigid',
    default_value=0
)

vis = ImageVisualiser(new_b50, cut=get_com(new_b50), figure_size_in=5)
vis.add_comparison_overlay(template_b50_to_new_rigid)
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/template_b50_to_new_b50_rigid.jpeg",dpi=400)

sitk.WriteImage(template_b50_to_new_rigid,"/home/alicja/PET_LAB_PROCESSED/test_rigid_reg_template_b50_to_new.nii.gz")

img_affine, tfm_img_affine,_=fast_symmetric_forces_demons_registration(
    new_b50,
    template_b50_to_new_rigid,
    resolution_staging=[4,2],
    iteration_staging=[3,3])
    
vis = ImageVisualiser(new_b50, cut=get_com(new_b50), figure_size_in=5)
vis.add_comparison_overlay(img_affine)
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/template_b50_to_new_b50_affine.jpeg",dpi=400)

sitk.WriteImage(img_affine,"/home/alicja/PET_LAB_PROCESSED/test_affine_reg_template_b50_to_new.nii.gz")