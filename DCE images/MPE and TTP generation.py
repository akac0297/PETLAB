#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import copy
import glob
import pandas
import datetime

def obtainDCEimages(pat_no="04",timept="1",path="/home/alicja/"):
    folder="PET_LAB_PROCESSED/WES_0"+pat_no+"/IMAGES/"
    baseline_image=sitk.ReadImage(path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_ACQ_0.nii.gz")
    baseline_image=sitk.Cast(baseline_image,sitk.sitkInt16)
    DCE_images=glob.glob(path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_ACQ_*.nii.gz")
    DCE_images=[image for image in DCE_images if "ACQ_0.nii.gz" not in image]
    return(baseline_image,DCE_images)

def returnSubImages(baseline_image,DCE_images):
    sub_images=[]
    for image in DCE_images:
        image=sitk.Cast(sitk.ReadImage(image),sitk.sitkInt16)
        new_image=image-baseline_image
        sub_images.append(new_image)
    return(sub_images)

def generateMPE(sub_images, pat_no, timept,path):
    stacked_arr = np.stack([sitk.GetArrayFromImage(i) for i in sub_images])
    max_arr_values = np.max(stacked_arr, axis=0)
    MPE_img = sitk.GetImageFromArray(max_arr_values)
    MPE_img.CopyInformation(sub_images[0])
    folder="PET_LAB_PROCESSED/WES_0"+pat_no+"/IMAGES/"
    sitk.WriteImage(MPE_img, path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_MPE_sub.nii.gz")
    return(MPE_img)

def obtainImageTimes(pat_no,timept):
    patient_data=pandas.read_csv(path+"PET_LAB_PROCESSED/PATIENT_DATA.csv")
    DCE_data=patient_data[patient_data["IMAGE_TYPE"].str.contains("MRI_T1W_DCE_ACQ")]
    DCE_data=(DCE_data[DCE_data["TIMEPOINT"]==timept])
    DCE_data=DCE_data[DCE_data["PATIENT_ID"]==int(pat_no)]
    time_dict={}

    baseline_info =DCE_data[DCE_data["IMAGE_TYPE"]=="MRI_T1W_DCE_ACQ_0"]
    baseline_time = baseline_info["TIME_HHMMSS"].values[0]
    time_dict.update({0:baseline_time})
    for item in range(1,6):
        image=DCE_data[DCE_data["IMAGE_TYPE"]=="MRI_T1W_DCE_ACQ_"+str(item)]
        val = image["TIME_HHMMSS"].values[0]
        key=item
        time_dict.update({key: val})
    
    baseline_time=str(time_dict.get(0))
    if len(baseline_time)==5:
        baseline_time="0"+baseline_time
    format = '%H%M%S'
    time_diff={}
    for timepoint in range(1,6):
        time = str(time_dict.get(timepoint))
        if len(time)==5:
            time="0"+time
        startDateTime = datetime.datetime.strptime(baseline_time, format)
        endDateTime = datetime.datetime.strptime(time, format)
        diff = endDateTime - startDateTime
        time_diff.update({timepoint: int(diff.total_seconds())})
    
    return(time_diff)

def generateTTP(sub_images,pat_no,timept,path):
    stacked_arr = np.stack([sitk.GetArrayFromImage(i) for i in sub_images])
    max_arr = np.argmax(stacked_arr, axis=0)+1
    np.unique(max_arr, return_counts=True)
    argmax_img=sitk.GetImageFromArray(max_arr)
    argmax_img.CopyInformation(sub_images[0])
    argmax_img = sitk.Cast(argmax_img, sitk.sitkInt16)
    
    TTP_arr=sitk.GetArrayFromImage(argmax_img)
    new_TTP_arr=copy.deepcopy(TTP_arr)
    TTP_vals=obtainImageTimes(pat_no,timept)    
    for array_idx in range(1,np.max(TTP_arr)+1):
        new_TTP_arr[TTP_arr==array_idx]=TTP_vals.get(array_idx)

    TTP_img=sitk.GetImageFromArray(new_TTP_arr)
    TTP_img.CopyInformation(sub_images[0])
    TTP_img=sitk.Cast(TTP_img, sitk.sitkInt16)

    folder="PET_LAB_PROCESSED/WES_0"+pat_no+"/IMAGES/"
    sitk.WriteImage(TTP_img, path+folder+"WES_0"+pat_no+"_TIMEPOINT_"+timept+"_MRI_T1W_DCE_TTP_sub.nii.gz")
    return(TTP_img)


def runDCEgeneration(path="/home/alicja/",patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]):
    MPE_images_generated=0
    TTP_images_generated=0
    timepoints=["1","2","3"]
    for pat_no in patient_list:
        for timept in timepoints:
            baseline_image,DCE_images = obtainDCEimages(pat_no=pat_no,timept=timept,path=path)
            sub_images=returnSubImages(baseline_image,DCE_images)
            MPE_img=generateMPE(sub_images,pat_no,timept,path)
            MPE_images_generated+=1
            print("Number of MPE images: ", MPE_images_generated)
            TTP_img=generateTTP(sub_images,pat_no,timept,path)
            TTP_images_generated+=1
            print("Number of TTP images: ", TTP_images_generated)
    
    return(MPE_images_generated,TTP_images_generated)

if __name__ == "__main__":

    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    path="/home/alicja/"
    
    MPE_images_generated,TTP_images_generated=runDCEgeneration(path=path)
    print("Total number MPE images:",MPE_images_generated,"Total number TTP images:", TTP_images_generated)