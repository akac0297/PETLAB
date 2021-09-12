#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import stats
import os

MRI_seg_path="/home/alicja/PET-LAB Code/PET-LAB/Old MRI segmentations/"
PET_seg_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/New PET tumours/"

PET_patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19"]
new_timepoints=["1","2","3"]

path="/home/alicja/PET_LAB_PROCESSED/"
MRI_timepoints={"04": ["4","5","6"],"05":["4","5","6"],"06":["4","5","6"],"07":["4","5","6"],"08":["4","5","6"],
"09":["6","7","8"],"10":["4","5","6"],"12":["4","5","6"],"13":["4","5","6"],"14":["4","5","6"],"15":["4","5","6"],"16":["3","4","5"],
"18":["4","5","6"],"19":["4","5"],"21":["2","3","4"],"23":["2","3","4"]}

def getPETImg(patient_no,timepoint,path):
    img=path+f"WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_{timepoint}_PET.nii.gz"
    return(img)

def getPETseg(patient_no,timepoint,PET_seg_path):
    PET_img=PET_seg_path+f"WES_0{patient_no}_TIMEPOINT_{timepoint}_PET_TUMOUR.nii.gz"
    return(PET_img)

def convertPETseg(PET_seg):
    PET_seg_arr=sitk.GetArrayFromImage(PET_seg)
    new_PET_arr=np.ones(np.shape(PET_seg_arr))
    new_PET_arr[PET_seg_arr==0]=0
    binary_PET_seg=sitk.GetImageFromArray(new_PET_arr)
    binary_PET_seg.CopyInformation(PET_seg)
    return(binary_PET_seg)

def getMRIseg(patient_no,timepoint,MRI_seg_path,img_type):
    MRI_img=MRI_seg_path+f"test_label_threshold_0{patient_no}_{timepoint}_{img_type}_hist.nii.gz"
    return(MRI_img)

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
    mad=stats.median_abs_deviation(tumour)
    p_5 = np.percentile(tumour, 5)
        
    featuresDict={"PATIENT_ID":patient_no, "TIMEPOINT":timepoint, "95%":p_95, "MEDIAN":median, "MEAN":mean, "STD DEV": sd,
    "IQR":iqr, "MEDIAN ABS DEV":mad, "5%":p_5}
    
    return(featuresDict)

def PETintensityAnalysis(img,tumour_seg,patient_no,timepoint):
    tumour_seg=sitk.Resample(tumour_seg, img)
    masked_seg=sitk.Mask(img,tumour_seg==1)
    
    tumour_arr=sitk.GetArrayFromImage(masked_seg)
    tumour=tumour_arr.flatten()
    tumour=tumour[tumour>0]

    p_95 = np.percentile(tumour, 95)
    SUV_max=np.max(tumour)
    median=np.median(tumour)
    mean=np.average(tumour)
    sd=np.std(tumour)
    iqr=np.subtract(*np.percentile(tumour, [75, 25]))
    mad=stats.median_abs_deviation(tumour)
    p_5 = np.percentile(tumour, 5)
        
    featuresDict={"PATIENT_ID":patient_no, "TIMEPOINT":timepoint, "95%":p_95, "MAX": SUV_max, "MEDIAN":median, "MEAN":mean, "STD DEV": sd,
    "IQR":iqr, "MEDIAN ABS DEV":mad, "5%":p_5}
    
    return(featuresDict)


def ADCanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% ADC","MEDIAN ADC","MEAN ADC","STD DEV ADC","IQR ADC", "MEDIAN ABS DEV ADC", "5% ADC", "IQR/MEDIAN ADC"])
    patient_idx=0
    for patient, timepoints in MRI_timepoints.items():
        patient_idx+=1
        folder="WES_0"+patient+"/IMAGES/"
        for i in range(len(timepoints)):
            old_timepoint=timepoints[i]
            new_timepoint=new_timepoints[i]
            ADC_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+new_timepoint+"_MRI_DWI_ADC.nii.gz")
            seg_filename=getMRIseg(patient,old_timepoint,MRI_seg_path,"B800T")
            seg=sitk.ReadImage(seg_filename)
            featuresDict = intensityAnalysis(ADC_img,seg,patient,new_timepoint)
            idx=3*patient_idx+i
            df.loc[idx]=[int(patient)]+[int(new_timepoint)]+[featuresDict.get("95%")]+[featuresDict.get("MEDIAN")]+[featuresDict.get("MEAN")]+[featuresDict.get("STD DEV")]+[featuresDict.get("IQR")]+[featuresDict.get("MEDIAN ABS DEV")]+[featuresDict.get("5%")]+[featuresDict.get("IQR")/featuresDict.get("MEDIAN")]
        print(f"Patient WES_0{patient} ADC analysis complete")
    return(df)

def MPEanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","MEDIAN ABS DEV MPE","5% MPE", "IQR/MEDIAN MPE"])
    patient_idx=0
    for patient, timepoints in MRI_timepoints.items():
        patient_idx+=1
        folder="WES_0"+patient+"/IMAGES/"
        for i in range(len(timepoints)):
            old_timepoint=timepoints[i]
            new_timepoint=new_timepoints[i]
            MPE_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+new_timepoint+"_MRI_T1W_DCE_MPE_sub.nii.gz")
            seg_filename=getMRIseg(patient,old_timepoint,MRI_seg_path,"MPE")
            seg=sitk.ReadImage(seg_filename)
            featuresDict = intensityAnalysis(MPE_img,seg,patient,new_timepoint)
            idx=3*patient_idx+i
            df.loc[idx]=[int(patient)]+[int(new_timepoint)]+[featuresDict.get("95%")]+[featuresDict.get("MEDIAN")]+[featuresDict.get("MEAN")]+[featuresDict.get("STD DEV")]+[featuresDict.get("IQR")]+[featuresDict.get("MEDIAN ABS DEV")]+[featuresDict.get("5%")]+[featuresDict.get("IQR")/featuresDict.get("MEDIAN")]
        print(f"Patient WES_0{patient} MPE analysis complete")
    return(df)

def TTPanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","MEDIAN ABS DEV TTP","5% TTP", "IQR/MEDIAN TTP"])
    patient_idx=0
    for patient, timepoints in MRI_timepoints.items():
        patient_idx+=1
        folder="WES_0"+patient+"/IMAGES/"
        for i in range(len(timepoints)):
            old_timepoint=timepoints[i]
            new_timepoint=new_timepoints[i]
            TTP_img=sitk.ReadImage(path+folder+"WES_0"+patient+"_TIMEPOINT_"+new_timepoint+"_MRI_T1W_DCE_TTP_sub.nii.gz")
            seg_filename=getMRIseg(patient,old_timepoint,MRI_seg_path,"MPE")
            seg=sitk.ReadImage(seg_filename)
            featuresDict = intensityAnalysis(TTP_img,seg,patient,new_timepoint)
            idx=3*patient_idx+i
            df.loc[idx]=[int(patient)]+[int(new_timepoint)]+[featuresDict.get("95%")]+[featuresDict.get("MEDIAN")]+[featuresDict.get("MEAN")]+[featuresDict.get("STD DEV")]+[featuresDict.get("IQR")]+[featuresDict.get("MEDIAN ABS DEV")]+[featuresDict.get("5%")]+[featuresDict.get("IQR")/featuresDict.get("MEDIAN")]
        print(f"Patient WES_0{patient} TTP analysis complete")
    return(df)

def SUVanalysis(path, PET_patient_list, new_timepoints, PET_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","TIMEPOINT","95% SUV","MAX SUV","MEDIAN SUV","MEAN SUV","STD DEV SUV","IQR SUV","MEDIAN ABS DEV SUV","5% SUV"])
    patient_idx=0
    for patient in PET_patient_list:
        patient_idx+=1
        for i in range(len(new_timepoints)):
            timepoint=new_timepoints[i]
            PET_img_filename=getPETImg(patient,timepoint,path)
            PET_seg_filename=getPETseg(patient,timepoint,PET_seg_path)
            idx=3*patient_idx+i
            if os.path.isfile(PET_img_filename):
                PET_img=sitk.ReadImage(PET_img_filename)
                PET_seg=sitk.ReadImage(PET_seg_filename)
                new_PET_seg=convertPETseg(PET_seg)
                featuresDict=PETintensityAnalysis(PET_img,new_PET_seg,patient,timepoint)
                df.loc[idx]=[int(patient)]+[int(timepoint)]+[featuresDict.get("95%")]+[featuresDict.get("MAX")]+[featuresDict.get("MEDIAN")]+[featuresDict.get("MEAN")]+[featuresDict.get("STD DEV")]+[featuresDict.get("IQR")]+[featuresDict.get("MEDIAN ABS DEV")]+[featuresDict.get("5%")]
            else:
                df.loc[idx]=[int(patient)]+[int(timepoint)]+[np.NaN]+[np.NaN]+[np.NaN]+[np.NaN]+[np.NaN]+[np.NaN]+[np.NaN]+[np.NaN]
        print(f"Patient WES_0{patient} SUV analysis complete")
    return(df)

"""
Running SUV functional analysis
"""
#test_PET_patient_list=["10","12","13"] # these don't currently work - I can't resample them properly
#SUV_df=SUVanalysis(path, PET_patient_list, new_timepoints, PET_seg_path)
#SUV_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_new.csv",index=False)

"""
Original running of MRI functional analysis code
"""
ADC_df=ADCanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path)
MPE_df=MPEanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path)
TTP_df=TTPanalysis(path, MRI_timepoints, new_timepoints, MRI_seg_path)
ADC_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_ADC_analysis_B800T_new.csv",index=False)
MPE_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_MPE_analysis_new2.csv",index=False)
TTP_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_TTP_analysis_new2.csv",index=False)

"""
Reading in MRI functional analysis dataframes separately and merging
"""
#ADC_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_ADC_analysis.csv")
#MPE_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_MPE_analysis.csv")
#TTP_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_TTP_analysis.csv")

DCE_df=pd.merge(MPE_df,TTP_df,on=["PATIENT_ID","TIMEPOINT"])
MRIFunctionalAnalysisDf=pd.merge(ADC_df,DCE_df,on=["PATIENT_ID","TIMEPOINT"])
MRIFunctionalAnalysisDf.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_functional_analysis_new2.csv",index=False)
# combining dataframes together: https://stackoverflow.com/questions/17978133/python-pandas-merge-only-certain-columns