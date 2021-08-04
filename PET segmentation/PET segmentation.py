#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration,
    transform_propagation,
    apply_field
)

patient_no="02"
timepoint="2"
ct="WES_002_2_20170825_CT_10_PETCT_WBHDIN_ONC_3.nii.gz"
pet="WES_002_2_20170825_PT_AC_4.nii.gz"
ct_plan="WES_002_3_20171023_CT_2.nii.gz"
image_ct_0=sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+ct)
image_pt_0_raw=sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+pet)
image_ct_plan = sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+ct_plan)
contour_breast_plan = sitk.ReadImage("/home/alicja/Documents/WES_002/STRUCTURES/WES_002_3_0_RTSTRUCT_CW_RIGHT_IMRT.nii.gz")
image_pt_0=sitk.Resample(image_pt_0_raw, image_ct_0)

def registerCTplantoCT(image_ct_0,image_ct_plan,visualise="T"):
    image_ct_plan_to_0_rigid, tfm_plan_to_0_rigid = initial_registration(
        image_ct_0,
        image_ct_plan,
        options={
            'shrink_factors': [8,4],
            'smooth_sigmas': [0,0],
            'sampling_rate': 0.5,
            'final_interp': 2,
            'metric': 'mean_squares',
            'optimiser': 'gradient_descent_line_search',
            'number_of_iterations': 25},
        reg_method='Rigid')
    
    if visualise=="T":
        vis = ImageVisualiser(image_ct_0, cut=[150,220,256], window=[-250, 500])
        vis.add_comparison_overlay(image_ct_plan_to_0_rigid)
        fig = vis.show()

    image_ct_plan_to_0_dir, tfm_plan_to_0_dir = fast_symmetric_forces_demons_registration(
        image_ct_0,
        image_ct_plan_to_0_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10]
    )

    if visualise=="T":
        vis = ImageVisualiser(image_ct_0, cut=[150,220,256], window=[-250, 500])
        vis.add_comparison_overlay(image_ct_plan_to_0_dir)
        fig = vis.show()

    return(image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir)

image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir=registerCTplantoCT(image_ct_0,image_ct_plan)

def registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint):
    contour_breast_plan_to_0_rigid = transform_propagation(
        image_ct_0,
        contour_breast_plan,
        tfm_plan_to_0_rigid,
        structure=True
    )

    contour_breast_plan_to_0_dir = apply_field(
        contour_breast_plan_to_0_rigid,
        tfm_plan_to_0_dir,
        structure=True
    )

    sitk.WriteImage(contour_breast_plan_to_0_dir,"PET_plan_breast_seg_"+patient_no+"_"+timepoint+".nii.gz")

    vis = ImageVisualiser(image_ct_0, axis='z', cut=get_com(contour_breast_plan_to_0_dir), window=[-250, 500])
    vis.add_scalar_overlay(image_pt_0, name='PET', colormap=plt.cm.magma, min_value=0.1, max_value=10000)
    vis.add_contour(contour_breast_plan_to_0_dir, name='BREAST', color='g')
    fig = vis.show()

    return(contour_breast_plan_to_0_dir)

contour_breast_plan_to_0_dir=registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint)

def getPETseg(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint):
    masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir)
    sitk.WriteImage(masked_pet_breast, "masked_pet_breast_WES_0" + patient_no + "_" + timepoint + ".nii.gz")

    masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
    mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
    mask_arr=mask_arr.flatten() 

    p = np.percentile(mask_arr[mask_arr>0], 97)
    tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)
    sitk.WriteImage(tum, "pet_seg_0"+patient_no+"_"+timepoint+"_97pc.nii.gz")

    return(masked_pet_breast,tum)

masked_pet_breast,tum=getPETseg(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint)