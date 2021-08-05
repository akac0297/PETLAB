#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import copy

#change this section once I have new data (function to obtain file names), this is just an example

pat_no="05"
q = 10
timept="4"

baseline_image=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_9.nii.gz", sitk.sitkInt16)

image_1=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_10.nii.gz", sitk.sitkInt16)
image_2=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_15.nii.gz", sitk.sitkInt16)
image_3=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_17.nii.gz", sitk.sitkInt16)
image_4=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_19.nii.gz", sitk.sitkInt16)
image_5=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/WES_005_4_20170717_MR_T1_FL3D_TRA_DYNAVIEWS_FL3D1_T1_FL3D_TRA_DYNAVIEWS_21.nii.gz", sitk.sitkInt16)

DCE_images=[image_1,image_2,image_3,image_4,image_5]

#code to generate MPE / TTP images

def returnSubImages(baseline_image,DCE_images):
    sub_images=[]

    for image_idx in DCE_images:
        new_image=DCE_images[image_idx]-baseline_image
        sub_images.append(new_image)
    
    return(sub_images)

def generateTTP(sub_images,pat_no,timept):
    stacked_arr = np.stack([sitk.GetArrayFromImage(i) for i in sub_images])
    max_arr = np.argmax(stacked_arr, axis=0)
    np.unique(max_arr, return_counts=True)
    argmax_img=sitk.GetImageFromArray(max_arr)
    argmax_img.CopyInformation(sub_images[0])
    argmax_img = sitk.Cast(argmax_img, sitk.sitkInt16)

    TTP_arr=sitk.GetArrayFromImage(argmax_img)
    new_TTP_arr=copy.deepcopy(TTP_arr)
    TTP_vals=[12,22,36,45,59] #need to obtain this from a dataframe of values (input is probably patient number and time point)

    for array_idx in range(0,np.max(TTP_arr)+1):
        new_TTP_arr[TTP_arr==array_idx]=TTP_vals[array_idx]

    TTP_img=sitk.GetImageFromArray(new_TTP_arr)
    TTP_img.CopyInformation(sub_images[0])
    TTP_img=sitk.Cast(TTP_img, sitk.sitkInt16)

    sitk.WriteImage(TTP_img, "TTP_Patient_"+pat_no+"_Timepoint_"+timept+".nii.gz")
    return(TTP_img)

def generateMPE(sub_images, pat_no, timept):
    stacked_arr = np.stack([sitk.GetArrayFromImage(i) for i in sub_images])
    max_arr_values = np.max(stacked_arr, axis=0)
    MPE_img = sitk.GetImageFromArray(max_arr_values)
    MPE_img.CopyInformation(sub_images[0])
    sitk.WriteImage(MPE_img, "MPE_Patient_"+pat_no+"_Timepoint_"+timept+".nii.gz")
    return(MPE_img)

# need to write a function to run generate TTP and generate MPE on all patients and all time points