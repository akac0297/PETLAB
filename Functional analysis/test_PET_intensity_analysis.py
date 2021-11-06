#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk

path="/home/alicja/PET_LAB_PROCESSED/"
PET_seg_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/"
PET_patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19"]
new_timepoints=["1","2","3"]

def getPETImg(patient_no,timepoint,path):
    img=path+f"WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_{timepoint}_PET.nii.gz"
    return(img)

def getPETseg(patient_no,timepoint,PET_seg_path):
    PET_img=PET_seg_path+f"WES_0{patient_no}_TIMEPOINT_{timepoint}_PET_TUMOUR_97_pc.nii.gz"
    return(PET_img)

def PETintensityAnalysis(img,tumour_seg,patient_no,timepoint):
    if tumour_seg.GetSize() != img.GetSize():
        print("resampling needed")
        tumour_seg=sitk.Resample(tumour_seg,img,sitk.Transform(),sitk.sitkNearestNeighbor)
        masked_seg=sitk.Mask(img,tumour_seg==1)
        tumour_arr=sitk.GetArrayFromImage(masked_seg)
    else:
        seg_arr=sitk.GetArrayFromImage(tumour_seg)
        img_arr=sitk.GetArrayFromImage(img)
        tumour_arr=img_arr[seg_arr==1]
    tumour=tumour_arr.flatten()
    print(np.min(tumour),np.max(tumour))
    tumour=tumour[tumour>0]

    p_95 = np.percentile(tumour, 95)
    median=np.median(tumour)
    mean=np.average(tumour)
    sd=np.std(tumour)
    iqr=np.subtract(*np.percentile(tumour, [75, 25]))
    p_5 = np.percentile(tumour, 5)
        
    featuresDict={"PATIENT_ID":patient_no, "TIMEPOINT":timepoint, "95%":p_95, "MEDIAN":median, "MEAN":mean, "STD DEV": sd,
    "IQR":iqr, "5%":p_5}
    
    return(featuresDict)

patient, timepoint = PET_patient_list[0],new_timepoints[0]


def convertPETseg(PET_seg):
    PET_seg_arr=sitk.GetArrayFromImage(PET_seg)
    new_PET_arr=np.ones(np.shape(PET_seg_arr))
    new_PET_arr[PET_seg_arr==0]=0
    binary_PET_seg=sitk.GetImageFromArray(new_PET_arr)
    binary_PET_seg.CopyInformation(PET_seg)
    return(binary_PET_seg)

def getFiles(patient,timepoint,path,convert=True):
    PET_img_filename=getPETImg(patient,timepoint,path)
    PET_seg_filename=getPETseg(patient,timepoint,PET_seg_path)
    PET_img=sitk.ReadImage(PET_img_filename)
    PET_seg=sitk.ReadImage(PET_seg_filename)
    if convert==True:
        PET_seg=convertPETseg(PET_seg)
    return(PET_img,PET_seg)

for patient in ["10","12","13"]:#PET_patient_list:
    for timepoint in new_timepoints:
        PET_img, PET_seg=getFiles(patient,timepoint,path)
        #print(PET_img.GetPixelIDTypeAsString())
        #print(PET_img.GetSize(),PET_img.GetSpacing(),PET_img.GetOrigin(),PET_img.GetDirection())
        #print(PET_seg.GetSize(),PET_seg.GetSpacing(),PET_seg.GetOrigin(),PET_seg.GetDirection())
        featuresDict=PETintensityAnalysis(PET_img,PET_seg,patient,timepoint)
        print(f"Patient {patient} time point {timepoint} PET intensity analysis complete")

PET_img,PET_seg=getFiles("12","2",path,True)
PET_seg_arr=sitk.GetArrayFromImage(PET_seg)
PET_img_arr=sitk.GetArrayFromImage(PET_img)
shape = np.shape(PET_img_arr)

img=getPETImg("12","2",path)
#PET_seg=sitk.Resample(PET_seg,PET_img)

#print(PET_img.GetOrigin(),PET_seg.GetOrigin())
#print(PET_img.GetSpacing(),PET_seg.GetSpacing())
print(PET_img.GetSpacing(),PET_seg.GetSpacing())
print(PET_img.GetSize(),PET_seg.GetSize())
#print(PET_img.GetDirection(),PET_seg.GetDirection())

PET_seg_arr=sitk.GetArrayFromImage(PET_seg)
PET_img_arr=sitk.GetArrayFromImage(PET_img)
#new_PET_arr=PET_img_arr[PET_seg_arr==1]
#print(np.shape(PET_seg_arr),np.shape(PET_img_arr))
print(np.max(PET_img_arr),np.max(PET_seg_arr))
#print(np.min(PET_img_arr[PET_seg_arr==1]),np.max(PET_img_arr[PET_seg_arr==1]))