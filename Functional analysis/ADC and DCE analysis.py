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
        
    featuresDict={"Patient":patient_no, "Timepoint":timepoint, "95%":p_95, "median":median, "mean":mean, "standard dev": sd,
    "IQR":iqr, "5%":p_5}
    series = pd.Series(featuresDict)
    
    return(series)

#need to create functions to run intensityAnalysis on all ADC, MPE, TTP images. Input is a list or dictionary of patient numbers and time points, and 
#output will be a dataframe of ADC, MPE, or TTP information for all patients and time points. Can also combine these dataframes together, but will
#need to add an additional column specifying the modality. Also note we can use the "masked pet breast" volume and SUVs to do analysis of the breast
#containing the tumour as well