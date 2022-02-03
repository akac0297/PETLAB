#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform
import numpy as np
import os

"""
Code to register the following images together using mutual information
- T2W (time point 1) to time point 2 and 3

Then after registering the images
- code to propagate the transforms (on the masks)
- after the masks are propagated, we split them into L and R breasts
"""

def readMRI(input_path,patient_id,timepoint):
    img_T2 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T2W.nii.gz", sitk.sitkFloat32)  
    return(img_T2)

def registerImages(img,new_img,tfm_path,patient_id,timepoint,img_type): 
    path=tfm_path+"WES_0"+patient_id+"_MRI_transforms/"
    img=sitk.Resample(img,new_img)

    img_rigid, tfm_img_rigid=linear_registration(
        new_img,
        img,
        shrink_factors = [8,4],
        smooth_sigmas = [0,0],
        sampling_rate = 0.5,
        final_interp = 2,
        reg_method = "affine",
        metric = 'mutual_information',
        optimiser= 'gradient_descent_line_search',
        number_of_iterations= 25)

    img_affine, tfm_img_affine,_=fast_symmetric_forces_demons_registration(
        new_img,
        img_rigid,
        resolution_staging=[4,2],
        iteration_staging=[3,3])

    sitk.WriteTransform(tfm_img_rigid,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_1_to_TIMEPOINT_"+timepoint+"_rigid.txt") 
    sitk.WriteTransform(tfm_img_affine,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_1_to_TIMEPOINT_"+timepoint+"_affine.txt")
    return(img_rigid, tfm_img_rigid, img_affine, tfm_img_affine)

def getMask(mask_path, patient_id, timepoint):
    mask=sitk.ReadImage(mask_path+f"WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT.nii.gz")
    return(mask)

def transformMask(img,new_img,patient_id,timepoint,img_type,tfm_path,mask_path,new_mask_path):
    mask=getMask(mask_path, patient_id, timepoint)
    img=sitk.Resample(img,new_img)
    tfm_img_rigid=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_1_to_TIMEPOINT_"+timepoint+"_rigid.txt") 
    tfm_img_affine=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_1_to_TIMEPOINT_"+timepoint+"_affine.txt")

    mask_rigid=apply_transform(
        mask,
        img,
        tfm_img_rigid
    )
    
    mask_affine = apply_transform(
        mask_rigid,
        img,
        tfm_img_affine
    )

    sitk.WriteImage(mask_affine,new_mask_path+f"WES_0{patient_id}_MRI_masks/WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W.nii.gz")
    return(mask_affine)

"""
Run registration and transform propagation to generate TP 2 and TP3 breast masks
"""

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]

input_path="/home/alicja/PET_LAB_PROCESSED/"
tfm_path="/home/alicja/PET-LAB Code/MRI_transforms/"
mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/Edited breast masks/"
new_mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/"
output_fp="/home/alicja/PET-LAB Code/BREAST_MASKS/T2W Breast masks/"

for patient_id in patient_list:
   img_T2_1=readMRI(input_path, patient_id, "1")
   img_T2_2=readMRI(input_path, patient_id, "2")
   img_T2_3=readMRI(input_path, patient_id, "3")
   os.makedirs(tfm_path+f"WES_0{patient_id}_MRI_transforms")
   os.makedirs(new_mask_path+f"WES_0{patient_id}_MRI_masks")
   img_rigid, tfm_img_rigid, img_affine, tfm_img_affine=registerImages(img_T2_1,img_T2_2,tfm_path,patient_id,"2","T2W")
   img_rigid, tfm_img_rigid, img_affine, tfm_img_affine=registerImages(img_T2_1,img_T2_3,tfm_path,patient_id,"3","T2W")
   mask=getMask(mask_path=mask_path, patient_id=patient_id, timepoint="1")
   mask_affine_2=transformMask(img_T2_1,img_T2_2,patient_id,"2","T2W",tfm_path,mask_path,new_mask_path)
   mask_affine_3=transformMask(img_T2_1,img_T2_3,patient_id,"3","T2W",tfm_path,mask_path,new_mask_path)

"""
After the masks have propagated, manually edit the T2w TP 2 and TP3 masks in Slicer using the paint and erase features
"""

"""
After the masks have been edited, run code to split them into L and R breasts
"""

def SeparateBreasts(mask):
    sag_coords = np.where(sitk.GetArrayFromImage(mask)==1)[2]
    cutoff = int(0.5*(sag_coords.min() + sag_coords.max()))
    print(cutoff)
    
    arr = sitk.GetArrayFromImage(mask)
    
    _, _, sag_indices = np.indices(arr.shape)
    
    arr = sitk.GetArrayFromImage(mask)
    arr[sag_indices>=cutoff] = 0
    mask_rightbreast = sitk.GetImageFromArray(arr)
    mask_rightbreast.CopyInformation(mask)

    arr = sitk.GetArrayFromImage(mask)
    arr[sag_indices<cutoff] = 0
    mask_leftbreast = sitk.GetImageFromArray(arr)
    mask_leftbreast.CopyInformation(mask)
    
    return(mask_leftbreast,mask_rightbreast)

def runT2WSeparation(mask_path,output_fp,patient_id,timepoint):
    filename=f"WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT"
    mask=sitk.ReadImage(mask_path+filename+".nii.gz")
    mask_leftbreast,mask_rightbreast=SeparateBreasts(mask)
    sitk.WriteImage(mask_leftbreast,output_fp+filename+"_L_breast.nii.gz")
    sitk.WriteImage(mask_rightbreast,output_fp+filename+"_R_breast.nii.gz")
    return("Separation complete")


"""Uncomment this code to run breast mask separation into L and R"""
#for i in range(len(patient_list)):
#    patient_id=patient_list[i]
#    for j in range(len(timepoints)):
#        timepoint=timepoints[j]
#        runT2WSeparation(mask_path,output_fp,patient_id,timepoint)