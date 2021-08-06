#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration,
    transform_propagation,
    apply_field
)

def getContourPlan(patient_no):
    df=pd.DataFrame(columns=["PATIENT_ID","CONTOUR_PLAN"])
    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    contour_plans=["WES_004_RTSIM_LABEL_CHESTWALL_LT_PTV.nii.gz",
                "WES_005_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_006_RTSIM_LABEL_CHEST_WALL_PTV.nii.gz",
                "WES_007_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_008_RTSIM_LABEL_CHESTWALL_PTV.nii.gz",
                "WES_009_RTSIM_LABEL_CHESTWALL_PTV.nii.gz",
                "WES_010_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_012_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_013_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_014_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_015_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_016_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_018_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_019_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_021_RTSIM_LABEL_CW_RIGHT_PTV.nii.gz",
                "WES_023_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz"
                ]
    for patient in patient_list:
        idx=patient_list.index(patient)
        df.loc[idx]=[int(patient)]+[contour_plans[patient_list.index(patient)]]
    test_df=df[df["PATIENT_ID"]==int(patient_no)]
    contour_plan = test_df["CONTOUR_PLAN"].values[0]
    return(contour_plan)

def getPETimages(patient_no,timepoint,path):
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    ct="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_CT_AC.nii.gz"
    pet="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET.nii.gz"
    ct_plan="WES_0"+patient_no+"_CT_RTSIM.nii.gz"
    image_ct_0=sitk.ReadImage(path+folder+ct)
    image_pt_0_raw=sitk.ReadImage(path+folder+pet)
    image_ct_plan = sitk.ReadImage(path+folder+ct_plan)
    contour_plan=getContourPlan(patient_no)
    contour_breast_plan = sitk.ReadImage(path+"PET_LAB_PROCESSED/WES_0"+patient_no+"/LABELS"+contour_plan)
    image_pt_0=sitk.Resample(image_pt_0_raw, image_ct_0)
    return(image_ct_0,image_pt_0_raw,image_pt_0,image_ct_plan,contour_breast_plan)

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

def maskPET(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,path):
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir)
    #sitk.WriteImage(masked_pet_breast, "masked_pet_breast_WES_0" + patient_no + "_" + timepoint + ".nii.gz")
    sitk.WriteImage(masked_pet_breast, path+folder+"WES_0" + patient_no + "_TIMEPOINT_" + timepoint + "_PET_IPSI_BREAST.nii.gz")
    
    masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
    return(masked_pet_breast)

def registerMasks(masked_pet_breast,patient_no,path,visualise="T"):
    #may need to mask PET / CT IMAGES and register those rather than trying to register masks together.
    #or find another way to convert a mask into an image.
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    mask1=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_1_PET_IPSI_BREAST.nii.gz")
    mask1=sitk.Resample(mask1,masked_pet_breast)
    image_mask1_to_0_rigid, tfm_mask1_to_0_rigid = initial_registration(
        masked_pet_breast,
        mask1,
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
        vis = ImageVisualiser(masked_pet_breast, cut=get_com(masked_pet_breast), window=[0,1])
        vis.add_comparison_overlay(image_mask1_to_0_rigid)
        fig = vis.show()

    image_mask1_to_0_dir, tfm_mask1_to_0_dir = fast_symmetric_forces_demons_registration(
        masked_pet_breast,
        image_mask1_to_0_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10]
    )

    if visualise=="T":
        vis = ImageVisualiser(masked_pet_breast, cut=get_com(masked_pet_breast), window=[0,1])
        vis.add_comparison_overlay(image_mask1_to_0_dir)
        fig = vis.show()
    return(image_mask1_to_0_rigid, tfm_mask1_to_0_rigid,image_mask1_to_0_dir,tfm_mask1_to_0_dir)

def maskWithTumour(path,patient_no,masked_pet_breast,tfm_mask1_to_0_rigid,tfm_mask1_to_0_dir):
    #this one also may not work
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    tum=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_1_PET_TUMOUR.nii.gz")
    tum_to_0_rigid = transform_propagation(
        masked_pet_breast,
        tum,
        tfm_mask1_to_0_rigid,
        structure=True
    )

    tum_to_0_dir = apply_field(
        tum_to_0_rigid,
        tfm_mask1_to_0_dir,
        structure=True
    )
    
    tum_dilate=sitk.BinaryDilate(tum_to_0_dir, (20,20,20))
    masked_pet_breast=sitk.Mask(masked_pet_breast,tum_dilate==1)
    return(masked_pet_breast)

def getPETseg(masked_pet_breast,image_pt_0_raw,patient_no,timepoint,path):
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
    mask_arr=mask_arr.flatten() 

    p = np.percentile(mask_arr[mask_arr>0], 98) #should this be the mask or the whole patient ???
    tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)
    tum = sitk.Cast(tum, sitk.sitkInt64)
    tum_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum))
    tum = (tum_cc==1)
    sitk.WriteImage(tum, path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET_TUMOUR.nii.gz")

    return(masked_pet_breast,tum)

#def getPETseg(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,path):
#    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
#    masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir)
#    sitk.WriteImage(masked_pet_breast, path+folder+"WES_0"+patient_no +"_TIMEPOINT_"+timepoint+"PET_IPSI_BREAST.nii.gz")

#    masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
#    mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
#    mask_arr=mask_arr.flatten() 

#    p = np.percentile(mask_arr[mask_arr>0], 98)
#    tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)
#    tum = sitk.Cast(tum, sitk.sitkInt64)
#    tum_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum))
#    tum = (tum_cc==1)
#    sitk.WriteImage(tum, path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET_TUMOUR.nii.gz")

#    return(masked_pet_breast,tum)

path="/home/alicja/"
patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]

#example
patient_no=patient_list[0]
timepoint=timepoints[1]

image_ct_0,image_pt_0_raw,image_pt_0,image_ct_plan,contour_breast_plan=getPETimages(patient_no,timepoint,path)
image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir=registerCTplantoCT(image_ct_0,image_ct_plan)
contour_breast_plan_to_0_dir=registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint)
masked_pet_breast=maskPET(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,path)
if (timepoint=="2" or timepoint=="3"):
    image_mask1_to_0_rigid, tfm_mask1_to_0_rigid,image_mask1_to_0_dir,tfm_mask1_to_0_dir=registerMasks(
        masked_pet_breast,patient_no,path,visualise="T")
    masked_pet_breast=maskWithTumour(path,patient_no,masked_pet_breast,tfm_mask1_to_0_rigid,tfm_mask1_to_0_dir)
masked_pet_breast,tum=getPETseg(masked_pet_breast,image_pt_0_raw,patient_no,timepoint,path)
