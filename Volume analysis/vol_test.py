#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
PET_timepoints=["1","2","3"]
MRI_timepoints={"04": ["4","5","6"],"05":["4","5","6"],"06":["4","5","6"],"07":["4","5","6"],"08":["4","5","6"],
"09":["6","7","8"],"10":["4","5","6"],"12":["4","5","6"],"13":["4","5","6"],"14":["4","5","6"],"15":["4","5","6"],"16":["3","4","5"],
"18":["4","5","6"],"19":["4","5"],"21":["2","3","4"],"23":["2","3","4"]}
path="/home/alicja/PET_LAB_PROCESSED/"

MRI_seg_path="/home/alicja/PET-LAB Code/PET-LAB/Old MRI segmentations/"
PET_seg_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/New PET tumours/"

def getPETImg(patient_no,timepoint,PET_seg_path):
    PET_img=PET_seg_path+f"WES_0{patient_no}_TIMEPOINT_{timepoint}_PET_TUMOUR.nii.gz"
    return(PET_img)

def getMRIImg(patient_no,timepoint,MRI_seg_path,img_type):
    MRI_img=MRI_seg_path+f"test_label_threshold_0{patient_no}_{timepoint}_{img_type}_hist.nii.gz"
    return(MRI_img)

def getPETVolume(patient_no,timepoint,PET_seg_path):
    PET_img=getPETImg(patient_no,timepoint,PET_seg_path)
    seg=sitk.ReadImage(PET_img)
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array>0)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    return(volume)

img_1=getPETImg("04","1",PET_seg_path)
img_2=getPETImg("04","2",PET_seg_path)
img_3=getPETImg("04","3",PET_seg_path)
#print(img_1,img_2,img_3)

img1=sitk.ReadImage(img_1)
img2=sitk.ReadImage(img_2)
img3=sitk.ReadImage(img_3)
img2=sitk.Resample(img2,img3)

arr1=sitk.GetArrayFromImage(img1)
arr2=sitk.GetArrayFromImage(img2)
arr3=sitk.GetArrayFromImage(img3)
#arr4=arr3*2-arr2+1
#print(np.min(arr4),np.max(arr4))

print(img1.GetSpacing(),img2.GetSpacing(),img3.GetSpacing())

sum1=np.sum(arr1>0)
sum2=np.sum(arr2>0)
sum3=np.sum(arr3>0)
#sum4=np.sum(arr4>1)
print(sum1,sum2,sum3)#,sum4)

vol_004_1=getPETVolume("04","1",PET_seg_path)
vol_004_2=getPETVolume("04","2",PET_seg_path)
vol_004_3=getPETVolume("04","3",PET_seg_path)

print(vol_004_1,vol_004_2,vol_004_3)