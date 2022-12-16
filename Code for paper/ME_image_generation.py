#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import glob

path="/home/alicja/PET_LAB_PROCESSED/"

def obtainDCEimages(pat_no="04",timept="1",path=path):
    folder="WES_0"+pat_no+"/IMAGES/"
    baseline_image=sitk.ReadImage(path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_ACQ_0.nii.gz")
    baseline_image=sitk.Cast(baseline_image,sitk.sitkFloat64)
    DCE_images=glob.glob(path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_ACQ_*.nii.gz")
    DCE_images=[image for image in DCE_images if "ACQ_0.nii.gz" not in image]
    return(baseline_image,DCE_images)

def returnNormImages(baseline_image,DCE_images):
    norm_images=[]
    for image in DCE_images:
        image=sitk.Cast(sitk.ReadImage(image),sitk.sitkFloat64)
        img_array=sitk.GetArrayFromImage(image)
        baseline_array=sitk.GetArrayFromImage(baseline_image)
        a=img_array-baseline_array
        b=baseline_array
        new_img_arr = 100*np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        new_img=sitk.GetImageFromArray(new_img_arr)
        new_img.CopyInformation(image)
        norm_images.append(new_img)
        print(f"image added to array")
    return(norm_images)

def generateME(norm_images, pat_no, timept,path):
    stacked_arr = np.stack([sitk.GetArrayFromImage(i) for i in norm_images])
    max_arr_values = np.max(stacked_arr, axis=0)
    ME_img = sitk.GetImageFromArray(max_arr_values)
    ME_img.CopyInformation(norm_images[0])
    folder="WES_0"+pat_no+"/IMAGES/"
    sitk.WriteImage(ME_img, path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_ME.nii.gz")
    return(ME_img)

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
def runDCEgeneration(path=path,patient_list=patient_list):
    ME_images_generated=0
    timepoints=["1","2","3"]
    for pat_no in patient_list:
        for timept in timepoints:
            baseline_image,DCE_images = obtainDCEimages(pat_no=pat_no,timept=timept,path=path)
            norm_images=returnNormImages(baseline_image,DCE_images)
            _=generateME(norm_images,pat_no,timept,path)
            ME_images_generated+=1
            print("Number of ME images: ", ME_images_generated)
    
    return(ME_images_generated)

ME_images_generated=runDCEgeneration(path=path,patient_list=patient_list)
print("Total number ME images:",ME_images_generated)