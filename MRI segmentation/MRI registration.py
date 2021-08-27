#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
from platipy.imaging.registration.registration import (
    initial_registration,
    transform_propagation,
    apply_field,
    fast_symmetric_forces_demons_registration)

from platipy.imaging.registration.linear import linear_registration #replaces "initial_registration"
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration

from platipy.imaging.registration.utils import apply_transform #replaces "transform_propagation" and "apply_field"

import numpy as np
import os

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

"""
Code to register the following images together using mutual information
or correlation:
- B50T to B800T (all time points)
- DCE 0 to DCE 1-5 (all time points)
- T2W time point 1 to 2,3 (not necessary)

- T2W (any time point) to any time point of:
    - ADC
    - B50T
    - DCE 0
    - MPE
    - TTP
    - T1w NFS
    - T2w SPAIR

Then after registering the images
- code to propagate the transforms (on the masks). Basically if we want to register T2W time point 2 to DCE 3 
time point 2, we want to transform T2W time point 2 to DCE 0 (time point 2), then propagate the transform 
using "DCE 0 to DCE 3" for that time point
- after the masks are propagated, we split them into L and R breasts
- sometimes we have to dilate the masks for some image types, and then we split them into L and R breasts
"""

"""
Use mutual information or correlation to register:
- B50T to B800T (all time points)
- DCE 0 to DCE 1-5 (all time points)
"""

"""
Step 1: 
- function to read in all relevant images for a given patient and time point (B50T, B800T, DCE 0-5)
- run function on all time points (just for 1 patient)
"""
#input_path="/home/alicja/PET_LAB_PROCESSED/"
#patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
#patient_id=patient_list[0]
#timepoints=["1","2","3"]
#timepoint=timepoints[0]

def readMRIs(input_path,patient_id,timepoint):
    img_B50 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_DWI_B50.nii.gz", sitk.sitkFloat32)
    img_B800 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_DWI_B800.nii.gz", sitk.sitkFloat32)
    img_DCE_0 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_0.nii.gz", sitk.sitkFloat32)
    img_DCE_1 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_1.nii.gz", sitk.sitkFloat32)
    img_DCE_2 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_2.nii.gz", sitk.sitkFloat32)
    img_DCE_3 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_3.nii.gz", sitk.sitkFloat32)
    img_DCE_4 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_4.nii.gz", sitk.sitkFloat32)
    img_DCE_5 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_5.nii.gz", sitk.sitkFloat32)
    
    img_ADC = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_DWI_ADC.nii.gz", sitk.sitkFloat32)
    img_MPE = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_MPE_sub.nii.gz", sitk.sitkFloat32)
    img_TTP = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_TTP_sub.nii.gz", sitk.sitkFloat32)
    img_T1_NFS = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_NFS.nii.gz", sitk.sitkFloat32)
    img_T2_SPAIR = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T2W_SPAIR.nii.gz", sitk.sitkFloat32)
    img_T2 = sitk.ReadImage(input_path+f"WES_0{patient_id}/IMAGES/WES_0{patient_id}_TIMEPOINT_{timepoint}_MRI_T2W.nii.gz", sitk.sitkFloat32)  
    
    return(img_B50,img_B800,img_DCE_0,img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5,img_ADC,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR,img_T2)

#img_B50,img_B800,img_DCE_0,img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5,img_ADC,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR,img_T2=readMRIs(input_path,patient_id,timepoint)

"""
Step 2:
- create function to register 2 images to each other using mutual information (and save the transform in a 
new folder for each patient "WES_004_MRI_transforms" and save it as "WES_004 B50 TP 1 to B800 TP 1")
- test on B50T to B800T for this 1 patient and 1 timepoint and visualise results
"""

#tfm_path="/home/alicja/PET-LAB Code/MRI_transforms/"

def registerImages(img,new_img,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="T"): 
    path=tfm_path+"WES_0"+patient_id+"_MRI_transforms/"
    img=sitk.Resample(img,new_img)

    img_rigid, tfm_img_rigid=initial_registration(
        new_img,
        img,
        options={
            'shrink_factors': [8,4],
            'smooth_sigmas': [0,0],
            'sampling_rate': 0.5,
            'final_interp': 2,
            'reg_method': "affine",
            'metric': 'mutual_information',
            'optimiser': 'gradient_descent_line_search',
            'number_of_iterations': 25},
        reg_method='Rigid')
    
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(img_rigid))
        vis.add_contour(img_rigid)
        fig = vis.show()
                
    img_affine, tfm_img_affine=fast_symmetric_forces_demons_registration(
        new_img,
        img_rigid,
        resolution_staging=[4,2],
        iteration_staging=[3,3])
        
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(img_affine))
        vis.add_contour(img_affine)
        fig = vis.show()
    
    sitk.WriteTransform(tfm_img_rigid,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_rigid.txt") 
    sitk.WriteTransform(tfm_img_affine,path+"WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_affine.txt")
    return(img_rigid, tfm_img_rigid, img_affine, tfm_img_affine)



#img_type="B50"
#new_img_type="B800"

#img_rigid, tfm_img_rigid, img_affine, tfm_img_affine=registerImages(img_B50,img_B800,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")


#sitk.WriteImage(img_affine,tfm_path+"test_affine_img_B50toB800_PT_004_TP1.nii.gz")
"""
Results are okay but there is some error. Not sure if the registration error is greater than the error in the
MRI's without registration (but it does seem to be)
"""

"""
Step 3:
- run the function to register DCE 0 to DCE 1-5 for this 1 patient and 1 time point (visualise a few results also)
- save in the mri transforms folder for the patient as "WES_004 DCE_0 TP 1 to DCE_X TP 1"
"""

#img_type="DCE_0"

#DCE_imgs=[img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5]
#for i in range(5):
#    new_img_type="DCE_"+str(i+1)
#    img_DCE=DCE_imgs[i]
#    img_rigid, tfm_img_rigid, img_affine, tfm_img_affine=registerImages(img_DCE_0,img_DCE,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")
#    sitk.WriteImage(img_affine,tfm_path+f"test_affine_img_DCE_0_to_{new_img_type}_PT_004_TP1.nii.gz")


"""
Step 4: 
- run mutual information registration to register T2W (time point 1) to time point 1 of:
    - ADC
    - B50T
    - DCE 0
    - MPE
    - TTP
    - T1w NFS
    - T2w SPAIR
- save the transforms in a folder with good names (e.g. "WES_004_MRI_transforms") and "T2W TP1 to ADC TP 1"
"""

#img_type="T2W"

#new_img_types=["ADC","B50","DCE_0","MPE","TTP","T1W_NFS","T2W_SPAIR"]
#new_imgs=[img_ADC,img_B50,img_DCE_0,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR]

#for i in range(7):
#    new_img_type=new_img_types[i]
#    new_img_T2=img_T2
#    new_img=new_imgs[i]
#    new_img_T2=sitk.Resample(new_img_T2,new_img)
#    _, _, _, _=registerImages(new_img_T2,new_img,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")

"""
Step 5:
- create a function to input the masks (T2W all time points). Give option to choose patient number and time point
- input all time points of T2w for the first patient
"""
#mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/Edited breast masks/"

def getMask(mask_path, patient_id, timepoint):
    mask=sitk.ReadImage(mask_path+f"WES_0{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT.nii.gz")
    return(mask)

"""
Step 6:
- write code to propagate the transforms (on the masks). If we want to register T2W time point 2 to DCE 3 
time point 2, we want to transform T2W time point 2 to DCE 0 (time point 2), then propagate the transform 
using "DCE 0 to DCE 3" for that time point
- test with T2W to B50T
"""

#new_mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/"

def transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="T"):
    mask=getMask(mask_path, patient_id, timepoint)
    img=sitk.Resample(img,new_img)
    tfm_img_rigid=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_rigid.txt") 
    tfm_img_affine=sitk.ReadTransform(tfm_path+"WES_0"+patient_id+"_MRI_transforms/WES_0"+patient_id+"_"+img_type+"_TIMEPOINT_"+timepoint+"_to_"+new_img_type+"_TIMEPOINT_"+timepoint+"_affine.txt")

    mask_rigid=transform_propagation(
        img,
        mask,
        tfm_img_rigid,
        structure=True
    )
    
    mask_affine = apply_field(
        mask_rigid,
        tfm_img_affine,
        structure=True
    )

    sitk.WriteImage(mask_affine,new_mask_path+f"WES_0{patient_id}_MRI_masks/WES_0{patient_id}_TIMEPOINT_{timepoint}_{new_img_type}.nii.gz")
    
    if visualise=="T":
        vis = ImageVisualiser(new_img, cut=get_com(mask_affine))
        vis.add_contour(mask_affine, name='BREAST', color='g')
        fig = vis.show()    

    return(mask_affine)


#img_type="T2W"
#new_img_type="B50"

#mask_affine=transformMask(img_T2,img_B50,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

"""
Step 7:
- propagate the transforms for all the images, generating 1 breast mask for each image type for this 1 patient
and 1 time point (save in a patient-specific folder)

- T2W (time point 1) to time point 1 of:
    - ADC
    - B50T
    - DCE 0
    - MPE
    - TTP
    - T1w NFS
    - T2w SPAIR
"""

#img_type="T2W"
#new_img_types=["ADC","B50","DCE_0","MPE","TTP","T1W_NFS","T2W_SPAIR"]
#new_imgs=[img_ADC,img_B50,img_DCE_0,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR]

#for i in range(7):
#    new_img_type=new_img_types[i]
#    new_img=new_imgs[i]
#    mask_affine=transformMask(img_T2,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")


"""
so far the results are:
- good for B50, B800 and TTP and quite good for DCE 1-5
- average for ADC, T1 NFS
- poor for DCE_0, MPE, T2 SPAIR. This could be because of the bright tissue outline (near skin) at the breast
boundary.

I could try dilating the masks for DCE_0/MPE and T2W SPAIR images individually?

I could also try a different registration method (but this would impact DCE1-5 which are really good)

Or I could not register the images as the errors in registration seem comparable to those of the movement between
MRI scans
"""

"""
Step 7.5:
- create a function to propagate the mask transforms for:
    - B50T to B800T
    - DCE 0 to DCE 1-5
- run the function for TP 1 patient WES_004
"""
#img_type="B50"
#new_img_type="B800"
#img=img_B50
#new_img=img_B800
#mask_affine=transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

#img_type="DCE_0"
#img=img_DCE_0
#new_img_types=["DCE_1","DCE_2","DCE_3","DCE_4","DCE_5"]
#new_imgs=[img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5]
#for i in range(5):
#    new_img_type=new_img_types[i]
#    new_img=new_imgs[i]
#    mask_affine=transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")


"""
Step 8:
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
    mask_rightbreast_cc = sitk.RelabelComponent(sitk.ConnectedComponent(mask_rightbreast))
    mask_rightbreast = (mask_rightbreast_cc==1)

    arr = sitk.GetArrayFromImage(mask)
    arr[sag_indices<cutoff] = 0
    mask_leftbreast = sitk.GetImageFromArray(arr)
    mask_leftbreast.CopyInformation(mask)
    mask_leftbreast_cc = sitk.RelabelComponent(sitk.ConnectedComponent(mask_leftbreast))
    mask_leftbreast = (mask_leftbreast_cc==1)
    
    return(mask_leftbreast,mask_rightbreast)

"""
- read in a mask from f"/home/alicja/PET-LAB Code/BREAST_MASKS/WES_0{patient_id}_MRI_masks/""
- run through a particular time point, and all image types (ADC, B50, B800, DCE_0 to _5, MPE, TTP, T1W_NFS, T2W_SPAIR)
- save the masks as L and R breasts
"""

#input_fp = f"/home/alicja/PET-LAB Code/BREAST_MASKS/WES_0{patient_id}_MRI_masks/"

def runSeparation(input_fp,patient_id,timepoint):
    img_types=["ADC", "B50", "B800", "DCE_0","DCE_1","DCE_2","DCE_3","DCE_4","DCE_5", "MPE", 
               "T1W_NFS", "T2W_SPAIR", "TTP"]
    for img_type in img_types:
        filename=f"WES_0{patient_id}_TIMEPOINT_{timepoint}_{img_type}"
        mask=sitk.ReadImage(input_fp+filename+".nii.gz")
        mask_leftbreast,mask_rightbreast=SeparateBreasts(mask)
        sitk.WriteImage(mask_leftbreast,input_fp+filename+"_L_breast.nii.gz")
        sitk.WriteImage(mask_rightbreast,input_fp+filename+"_R_breast.nii.gz")
    return("Separation complete")

#runSeparation(input_fp,patient_id,timepoint)

"""
Step 9:
- clean up code so it can be run on any patient and any time point
- run the code on all patients and all time points
"""
"""
e.g. to run on all time points for patient WES_004:

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
patient_id=patient_list[0]
timepoints=["1","2","3"]
timepoint=timepoints[0]
"""

def registerMasks(patient_id,timepoint,input_path,tfm_path,mask_path,new_mask_path):
    input_fp = new_mask_path+f"WES_0{patient_id}_MRI_masks/"

    img_B50,img_B800,img_DCE_0,img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5,img_ADC,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR,img_T2=readMRIs(input_path,patient_id,timepoint)

    img_type="B50"
    new_img_type="B800"

    _, _, img_affine, _=registerImages(img_B50,img_B800,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")

    img_type="DCE_0"

    DCE_imgs=[img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5]
    for i in range(5):
        new_img_type="DCE_"+str(i+1)
        img_DCE=DCE_imgs[i]
        _, _, img_affine, _=registerImages(img_DCE_0,img_DCE,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")
        sitk.WriteImage(img_affine,tfm_path+f"test_affine_img_DCE_0_to_{new_img_type}_PT_004_TP1.nii.gz")

    img_type="T2W"
    new_img_types=["ADC","B50","DCE_0","MPE","TTP","T1W_NFS","T2W_SPAIR"]
    new_imgs=[img_ADC,img_B50,img_DCE_0,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR]

    for i in range(7):
        new_img_type=new_img_types[i]
        new_img_T2=img_T2
        new_img=new_imgs[i]
        new_img_T2=sitk.Resample(new_img_T2,new_img)
        _, _, _, _=registerImages(new_img_T2,new_img,tfm_path,patient_id,timepoint,img_type,new_img_type,visualise="F")

    img_type="T2W"
    new_img_types=["ADC","B50","DCE_0","MPE","TTP","T1W_NFS","T2W_SPAIR"]
    new_imgs=[img_ADC,img_B50,img_DCE_0,img_MPE,img_TTP,img_T1_NFS,img_T2_SPAIR]

    for i in range(7):
        new_img_type=new_img_types[i]
        new_img=new_imgs[i]
        _=transformMask(img_T2,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

    img_type="B50"
    new_img_type="B800"
    img=img_B50
    new_img=img_B800
    _=transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

    img_type="DCE_0"
    img=img_DCE_0
    new_img_types=["DCE_1","DCE_2","DCE_3","DCE_4","DCE_5"]
    new_imgs=[img_DCE_1,img_DCE_2,img_DCE_3,img_DCE_4,img_DCE_5]
    for i in range(5):
        new_img_type=new_img_types[i]
        new_img=new_imgs[i]
        _=transformMask(img,new_img,patient_id,timepoint,img_type,new_img_type,tfm_path,mask_path,new_mask_path,visualise="F")

    runSeparation(input_fp,patient_id,timepoint)
    return("Registration complete")

"""
Step 9.5 
- dilate the breast masks that need it and separate them into L and R
"""

def dilateMask(new_mask_path,patient_id,timepoint,img_type,dilation_width=(3,3,3)):
    mask_affine=sitk.ReadImage(new_mask_path+f"WES_0{patient_id}_MRI_masks/WES_0{patient_id}_TIMEPOINT_{timepoint}_{img_type}.nii.gz")
    dilated_mask=sitk.BinaryDilate(mask_affine,dilation_width)
    dilated_mask.CopyInformation(mask_affine)
    sitk.WriteImage(dilated_mask,new_mask_path+f"WES_0{patient_id}_MRI_masks/WES_0{patient_id}_TIMEPOINT_{timepoint}_{img_type}_dilate.nii.gz")
    return(dilated_mask)

def runDilatedSeparation(input_fp,patient_id,timepoint,img_types):
    for img_type in img_types:
        filename=f"WES_0{patient_id}_TIMEPOINT_{timepoint}_{img_type}_dilate"
        mask=sitk.ReadImage(input_fp+filename+".nii.gz")
        mask_leftbreast,mask_rightbreast=SeparateBreasts(mask)
        sitk.WriteImage(mask_leftbreast,input_fp+filename+"_L_breast.nii.gz")
        sitk.WriteImage(mask_rightbreast,input_fp+filename+"_R_breast.nii.gz")
    return("Separation complete")

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
            img_types=["DCE_0","MPE", "T1W_NFS", "T2W_SPAIR"]
            img_type=img_types[0]
            input_fp=new_mask_path+f"WES_0{patient_id}_MRI_masks/"
            for img_type in img_types:
                if img_type=="T2W_SPAIR":
                    _=dilateMask(new_mask_path,patient_id,timepoint,img_type, (5,5,3))
                else:
                    _=dilateMask(new_mask_path,patient_id,timepoint,img_type, (3,3,3))
            runDilatedSeparation(input_fp,patient_id,timepoint,img_types)
        print(f"WES_0{patient_id} mask registration complete")
    return("all mask registrations complete")

patient_list=["04"]
timepoints=["2","3"]
img_types=["DCE_0","MPE", "T1W_NFS", "T2W_SPAIR"]

input_path="/home/alicja/PET_LAB_PROCESSED/"
tfm_path="/home/alicja/PET-LAB Code/MRI_transforms/"
mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/Edited breast masks/"
new_mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/"

runAllMaskRegistrations(patient_list,timepoints,input_path,tfm_path,mask_path,new_mask_path)