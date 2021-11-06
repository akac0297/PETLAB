#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com
from platipy.imaging.registration.linear import linear_registration #replaces "initial_registration"
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform #replaces "transform_propagation" and "apply_field"
import numpy as np
import os

"""
Code to register the following images together using mutual information
- T2W (any time point) to any time point of:
    - B50T
    - DCE 1

Then after registering the images
- code to propagate the transforms (on the masks)
- after the masks are propagated, we split them into L and R breasts
"""

"""
Step 1: 
- function to read in all relevant images for a given patient and time point (B50T, DCE 1)
"""
def readMRIs(input_path,patient_id,timepoint):
    img_B50 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_DWI_B50.nii.gz", sitk.sitkFloat32)
    img_DCE_1 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_1.nii.gz", sitk.sitkFloat32)
    img_T2 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T2W.nii.gz", sitk.sitkFloat32)  
    
    return(img_B50,img_DCE_1,img_T2)
"""
Step 2:
- create function to register 2 images to each other using mutual information (and save the transform in a 
new folder for each patient "WES_004_MRI_transforms" and save it as "WES_004 B50 TP 1 to B800 TP 1")
- test on B50T to B800T for this 1 patient and 1 timepoint and visualise results
"""
def registerImages(img,new_img,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="T"): 
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
    
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(img_rigid))
        vis.add_contour(img_rigid)
        _ = vis.show()

    img_affine, tfm_img_affine,_=fast_symmetric_forces_demons_registration(
        new_img,
        img_rigid,
        resolution_staging=[4,2],
        iteration_staging=[3,3])
        
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(img_affine))
        vis.add_contour(img_affine)
        _ = vis.show()

    sitk.WriteTransform(tfm_img_rigid,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_rigid.txt") 
    sitk.WriteTransform(tfm_img_affine,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_affine.txt")
    return(img_rigid, tfm_img_rigid, img_affine, tfm_img_affine)

"""
Step 3:
- create a function to input the masks (T2W all time points). Give option to choose patient number and time point
- input all time points of T2w for the first patient
"""
def getMask(mask_path, patient_id, timepoint):
    mask=sitk.ReadImage(mask_path+f"WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT.nii.gz")
    return(mask)

"""
Step 4:
- write code to propagate the transforms (on the masks). Register T2W to DCE 1 and B50T
"""
def transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="T"):
    mask=getMask(mask_path, patient_id, timepoint)
    img=sitk.Resample(img,new_img)
    tfm_img_rigid=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_rigid.txt") 
    tfm_img_affine=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_affine.txt")

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

    sitk.WriteImage(mask_affine,new_mask_path+f"WES_0{patient_id}_MRI_masks/WES_0{patient_id}_TIMEPOINT_{timepoint}_{new_img_type}.nii.gz")
    
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(mask_affine))
        vis.add_contour(mask_affine, name='BREAST', color='g')
        fig = vis.show()    

    return(mask_affine)

"""
Step 5:
- after the masks are propagated, copy code to split them into L and R breasts
- run this code on all registered masks and save them in the MRI masks folder. Label them with L or R, image type, 
and patient and time point.
- visualise the masks to make sure they look decent (it's okay if there are some errors)
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

def runSeparation(input_fp,patient_id,timepoint):
    img_types=["B50","DCE_1"]
    for img_type in img_types:
        filename=f"WES_0{patient_id}_TIMEPOINT_{timepoint}_{img_type}"
        mask=sitk.ReadImage(input_fp+filename+".nii.gz")
        mask_leftbreast,mask_rightbreast=SeparateBreasts(mask)
        sitk.WriteImage(mask_leftbreast,input_fp+filename+"_L_breast.nii.gz")
        sitk.WriteImage(mask_rightbreast,input_fp+filename+"_R_breast.nii.gz")
    return("Separation complete")

"""
Step 6:
- run the code on all patients and all time points
"""
def registerMasks(patient_id,timepoint,input_path,tfm_path,mask_path,new_mask_path):
    input_fp = new_mask_path+f"WES_0{patient_id}_MRI_masks/"

    img_B50,img_DCE_1,img_T2=readMRIs(input_path,patient_id,timepoint)

    img_type="T2W"
    new_img_types=["B50","DCE_1"]
    new_imgs=[img_B50,img_DCE_1]

    for i in range(2):
        new_img_type=new_img_types[i]
        new_img_T2=img_T2
        new_img=new_imgs[i]
        new_img_T2=sitk.Resample(new_img_T2,new_img)
        _, _, _, _=registerImages(new_img_T2,new_img,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")

    img_type="T2W"
    new_img_types=["B50","DCE_1"]
    new_imgs=[img_B50,img_DCE_1]

    for i in range(2):
        new_img_type=new_img_types[i]
        new_img=new_imgs[i]
        _=transformMask(img_T2,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

    runSeparation(input_fp,patient_id,timepoint)
    return("Registration complete")

"""
This is code to run all the above on all patients and all time points.
"""
def runAllMaskRegistrations(patient_list,timepoints,input_path,tfm_path,mask_path,new_mask_path):
    for i in range(len(patient_list)):
        patient_id=patient_list[i]
        os.makedirs(tfm_path+f"WES_0{patient_id}_MRI_transforms")
        os.makedirs(new_mask_path+f"WES_0{patient_id}_MRI_masks")

        for j in range(len(timepoints)):
            timepoint=timepoints[j]
            registerMasks(patient_id,timepoint,input_path,tfm_path,mask_path,new_mask_path)
        print(f"WES_0{patient_id} mask registration complete")
    return("all mask registrations complete")

patient_list=["05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]

input_path="/home/alicja/PET_LAB_PROCESSED/"
tfm_path="/home/alicja/PET-LAB Code/MRI_transforms/"
mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/Edited breast masks/"
new_mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/"
output_fp="/home/alicja/PET-LAB Code/BREAST_MASKS/T2W Breast masks/"

#runAllMaskRegistrations(patient_list,timepoints,input_path,tfm_path,mask_path,new_mask_path)

def runT2WSeparation(mask_path,output_fp,patient_id,timepoint):
    filename=f"WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT"
    mask=sitk.ReadImage(mask_path+filename+".nii.gz")
    mask_leftbreast,mask_rightbreast=SeparateBreasts(mask)
    sitk.WriteImage(mask_leftbreast,output_fp+filename+"_L_breast.nii.gz")
    sitk.WriteImage(mask_rightbreast,output_fp+filename+"_R_breast.nii.gz")
    return("Separation complete")

for i in range(len(patient_list)):
    patient_id=patient_list[i]
    for j in range(len(timepoints)):
        timepoint=timepoints[j]
        runT2WSeparation(mask_path,output_fp,patient_id,timepoint)