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

#def getPETRadius(patient_no,timepoint,PET_seg_path):
#    volume=getPETVolume(patient_no,timepoint,PET_seg_path)
#    radius=np.cbrt(3*volume/(4*np.pi))
#    return(radius)

def getMRIVolume(patient_no,timepoint,MRI_seg_path,img_type):
    MRI_img=getMRIImg(patient_no,timepoint,MRI_seg_path,img_type)
    seg=sitk.ReadImage(MRI_img)
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array==1)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    return(volume)

#def getMRIRadius(patient_no,timepoint,MRI_seg_path,img_type):
#    volume=getMRIVolume(patient_no,timepoint,MRI_seg_path,img_type)
#    radius=np.cbrt(3*volume/(4*np.pi))
#    return(radius)

def getVolDF(patient_list,PET_timepoints,MRI_timepoints,PET_seg_path,MRI_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","IMAGE_TYPE","TIMEPOINT","TUMOUR VOLUME_CM3"])#,"RADIUS_CM"])
    MRI_img_types=["MPE","T2w","B50T","B800T"]
    for patient_no in patient_list:
        offset=(patient_list.index(patient_no))*15
        for timepoint in PET_timepoints:
            PET_img=getPETImg(patient_no,timepoint,PET_seg_path)
            if os.path.isfile(PET_img):
                PET_vol=getPETVolume(patient_no,timepoint,PET_seg_path)
                #PET_radius=getPETRadius(patient_no,timepoint,PET_seg_path)
            else:
                PET_vol=np.NaN
                #PET_radius=np.NaN
            tp_idx=PET_timepoints.index(timepoint)*5
            idx=offset+tp_idx
            print(idx)
            df.loc[idx]=[int(patient_no)]+["PET"]+[int(timepoint)]+[PET_vol]#+[PET_radius]
            if len(MRI_timepoints.get(patient_no))==3:
                MRI_timepoint=MRI_timepoints.get(patient_no)[int(timepoint)-1]
            elif len(MRI_timepoints.get(patient_no))==2:
                try:
                    MRI_timepoint=MRI_timepoints.get(patient_no)[int(timepoint)-1]
                except:
                    MRI_timepoint="3"
            for img_type in MRI_img_types:
                idx+=1
                MRI_img=getMRIImg(patient_no,MRI_timepoint,MRI_seg_path,img_type)
                if os.path.isfile(MRI_img):
                    MRI_vol=getMRIVolume(patient_no,MRI_timepoint,MRI_seg_path,img_type)
                    #MRI_radius=getMRIRadius(patient_no,MRI_timepoint,MRI_seg_path,img_type)
                else:
                    MRI_vol=np.NaN
                    #MRI_radius=np.NaN
                df.loc[idx]=[int(patient_no)]+[img_type+" MRI"]+[int(timepoint)]+[MRI_vol]#+[MRI_radius]
                print(idx)
    return(df)

df=getVolDF(patient_list,PET_timepoints,MRI_timepoints,PET_seg_path,MRI_seg_path)
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volume_analysis_new.csv",index=False)