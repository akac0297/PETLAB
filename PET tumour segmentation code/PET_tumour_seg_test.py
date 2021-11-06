#!/usr/bin/env python
# coding: utf-8
"""
Import modules
"""
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.registration.utils import apply_linear_transform

image_ct_0=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_CT_AC.nii.gz")
image_pt_0_raw=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_PET.nii.gz")
image_ct_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_CT_RTSIM.nii.gz")
contour_breast_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/LABELS/WES_004_RTSIM_LABEL_CHESTWALL_LT_CTV.nii.gz")

image_ct_0=sitk.Resample(image_ct_0,image_ct_plan)
image_pt_0=sitk.Resample(image_pt_0_raw,image_ct_0)

breast_plan=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/contour_breast_plan_PET_004.nii.gz")
#breast_plan_dilate=sitk.BinaryDilate(breast_plan,(3,3,3))
#sitk.WriteImage(breast_plan_dilate,"/home/alicja/PET_LAB_PROCESSED/dilated_PET_breast_004.nii.gz")

breast_plan_dilate=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/dilated_PET_breast_004.nii.gz")

breast_plan_dilate=sitk.Resample(breast_plan_dilate,image_ct_0)

#vis = ImageVisualiser(image_ct_0, axis='z', cut=get_com(breast_plan_dilate), window=[-250, 500])
#vis.add_scalar_overlay(image_pt_0, name='PET SUV', colormap=plt.cm.magma, min_value=0.1, max_value=10000)
#vis.add_contour(breast_plan_dilate, name='R BREAST', color='g')
#fig = vis.show()
#fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/PET_masked_contour_dilate.jpeg",dpi=400)

PET_breast=sitk.Mask(image_pt_0,breast_plan_dilate)
#sitk.WriteImage(PET_breast,"/home/alicja/PET_LAB_PROCESSED/PET_breast_masked.nii.gz")

def getPETseg(PET_breast,image_pt_0):
    mask_arr=sitk.GetArrayFromImage(PET_breast)
    mask_arr=mask_arr.flatten() 

    p = np.percentile(mask_arr[mask_arr>0], 98)
    print("percentile: ", p)
    tum = sitk.Mask(image_pt_0, PET_breast>p)
    tum = sitk.Cast(tum, sitk.sitkInt64)
    tum_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum))
    tum = (tum_cc==1)
    sitk.WriteImage(tum, "/home/alicja/PET_LAB_PROCESSED/PET_TUMOUR_test.nii.gz")

    return(tum)

tum = getPETseg(PET_breast,image_pt_0)
