#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk
import pandas as pd

patient_no="03"
timepoint="2"
filename="WES_003_2_20170207_MR_RESOLVE_DIFF_TRA_SPAIR_P2_RE_B50_800_RESOLVE_DIFF_TRA_SPAIR_P2_ADC_12.nii.gz"

img=sitk.ReadImage("/home/alicja/Documents/WES_0"+patient_no+"/IMAGES/"+filename)
tumour_seg=sitk.ReadImage("/home/alicja/Downloads/new_seg_0"+patient_no+"_"+timepoint+"_mri.nii.gz")

def intensityAnalysis(img,tumour_seg,patient_no,timepoint):
    tumour_seg=sitk.Resample(tumour_seg, img)
    masked_seg=sitk.Mask(img,tumour_seg==1)
    
    tumour_arr=sitk.GetArrayFromImage(masked_seg)
    tumour=tumour_arr.flatten()
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

path="/home/alicja/"

def ADCanalysis(path=path):
    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    timepoints=["1","2","3"]
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% ADC","MEDIAN ADC","MEAN ADC","STD DEV ADC","IQR ADC","5% ADC"])
    for patient in patient_list:
        folder="PET_LAB_PROCESSED/WES_0"+patient+"/IMAGES/"
        for timepoint in timepoints:
            ADC_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+timepoint+"MRI_DWI_ADC.nii.gz")
            seg=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_MRI_DCE_TUMOUR.nii.gz")
            featuresDict = intensityAnalysis(ADC_img,seg,patient,timepoint)
            idx=timepoints.index(str(timepoint)+str(patient))
            df.loc[idx]=[int(patient)]+[int(timepoint)]+[featuresDict.get("95% ADC")]+[featuresDict.get("MEDIAN ADC")]+[featuresDict.get("MEAN ADC")]+[featuresDict.get("STD DEV ADC")]+[featuresDict.get("IQR ADC")]+[featuresDict.get("5% ADC")]
    return(df)

def MPEanalysis(path=path):
    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    timepoints=["1","2","3"]
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","5% MPE"])
    for patient in patient_list:
        folder="PET_LAB_PROCESSED/WES_0"+patient+"/IMAGES/"
        for timepoint in timepoints:
            MPE_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+timepoint+"MRI_T1W_DCE_MPE_sub.nii.gz")
            seg=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_MRI_DCE_TUMOUR.nii.gz")
            featuresDict = intensityAnalysis(MPE_img,seg,patient,timepoint)
            idx=timepoints.index(str(timepoint)+str(patient))
            df.loc[idx]=[int(patient)]+[int(timepoint)]+[featuresDict.get("95% MPE")]+[featuresDict.get("MEDIAN MPE")]+[featuresDict.get("MEAN MPE")]+[featuresDict.get("STD DEV MPE")]+[featuresDict.get("IQR MPE")]+[featuresDict.get("5% MPE")]
    return(df)

def TTPanalysis(path=path):
    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    timepoints=["1","2","3"]
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","5% TTP"])
    for patient in patient_list:
        folder="PET_LAB_PROCESSED/WES_0"+patient+"/IMAGES/"
        for timepoint in timepoints:
            TTP_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+timepoint+"MRI_T1W_DCE_TTP_sub.nii.gz")
            seg=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_MRI_DCE_TUMOUR.nii.gz")
            featuresDict = intensityAnalysis(TTP_img,seg,patient,timepoint)
            idx=timepoints.index(str(timepoint)+str(patient))
            df.loc[idx]=[int(patient)]+[int(timepoint)]+[featuresDict.get("95% TTP")]+[featuresDict.get("MEDIAN TTP")]+[featuresDict.get("MEAN TTP")]+[featuresDict.get("STD DEV TTP")]+[featuresDict.get("IQR TTP")]+[featuresDict.get("5% TTP")]
    return(df)

ADC_df=ADCanalysis(path=path)
MPE_df=MPEanalysis(path=path)
TTP_df=TTPanalysis(path=path)

DCE_df=pd.merge(MPE_df,TTP_df[["PATIENT_ID","95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","5% TTP"]],on="PATIENT_ID",how="left")
FunctionalAnalysisDf=pd.merge(ADC_df,DCE_df[["PATIENT_ID","95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","5% MPE","95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","5% TTP"]],on="PATIENT_ID",how="left")

#need to combine these dataframes together: https://stackoverflow.com/questions/17978133/python-pandas-merge-only-certain-columns