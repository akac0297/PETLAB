#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import pandas as pd

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]
path="/home/alicja/"

#we want the segmentations to be in format 'WES_0XX_TIMEPOINT_Y_MRI_DCE_TUMOUR.nii.gz'
#or 'WES_0XX_TIMEPOINT_Y_PET_TUMOUR.nii.gz'

def getPETVolume(patient_no,timepoint,path):
    folder="/PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    seg=sitk.ReadImage("pet_seg_0"+patient_no+"_"+timepoint+"_97pc.nii.gz")
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array>0)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    return(volume)

def getPETRadius(patient_no,timepoint,path):
    volume=getPETVolume(patient_no,timepoint,path)
    radius=np.cbrt(3*volume/(4*np.pi))
    return(radius)

def getMRIVolume(patient_no,timepoint,path):
    folder="/PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    seg=sitk.ReadImage("new_seg_0"+patient_no+"_"+timepoint+"_mri.nii.gz")
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array==1)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    return(volume)

def getMRIRadius(patient_no,timepoint,path):
    volume=getMRIVolume(patient_no,timepoint,path)
    radius=np.cbrt(3*volume/(4*np.pi))
    return(radius)

patient_dict={"04":["4","5","6"],"05":["4","5","6"],"06":["4","5","6"],"07":["4","5","6"],
              "08":["4","5","6"],"09":["6","7","8"],"10":["4","5","6"],"12":["4","5","6"],"13":["4","5","6"],
              "14":["4","5","6"],"15":["4","5","6"],"16":["3","4","5"],"18":["4","5","6"],"19":["3","4","5"],
              "21":["2","3","4"],"23":["2","3","4"],"24":["3","4","5"]
             }

def getMRIVolandRad(timepoints,path,patient_dict):
    df=pd.DataFrame(columns=["PATIENT_ID","IMAGE_TYPE","TIMEPOINT","TUMOUR VOLUME_CM3","RADIUS_CM"])
    patient_list=list(patient_dict.keys())
    timepoints=[]
    for patient in patient_list:
        for timepoint in patient_dict.get(patient):
            timepoints.append(str(timepoint)+str(patient))
            MRI_vol=getMRIVolume(patient,timepoint,path)
            MRI_radius=getMRIRadius(patient,timepoint,path)
            idx=timepoints.index(str(timepoint)+str(patient))
            df.loc[idx]=[int(patient)]+["MRI"]+[int(timepoint)]+[MRI_vol]+[MRI_radius]
    return(df)

df=getMRIVolandRad(timepoints,path,patient_dict)