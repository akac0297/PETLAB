#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import stats
import os

"""
This code isn't finished, I just ended up finding IQR/Median in Excel for SUV and in the ADC_DCE_SUV code for MRI
"""


MRI_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_functional_analysis_new.csv")
SUV_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_new.csv")
SUV_functional_data=SUV_functional_data.drop([39,40,41])
timepoints=[1.0,2.0,3.0]


def getFeatureGroups(feature,timepoints,functional_data):
    subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
    tp1=subset1[feature].to_list()

    subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
    tp2=subset2[feature].to_list()

    subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
    tp3=subset3[feature].to_list()

    return(tp1,tp2,tp3)

def getIQRoverMedian(image_type,timepoints,functional_data):
    tp1_IQR,tp2_IQR,tp3_IQR=getFeatureGroups(f"IQR {image_type}",timepoints,functional_data)
    tp1_med,tp2_med,tp3_med=getFeatureGroups(f"MEDIAN {image_type}",timepoints,functional_data)
    #print(tp1_IQR,tp1_med)
    tp1_vals=[i / j for i, j in zip(tp1_IQR, tp1_med)]
    tp2_vals=[i / j for i, j in zip(tp2_IQR, tp2_med)]
    tp3_vals=[i / j for i, j in zip(tp3_IQR, tp3_med)]
    return(tp1_vals,tp2_vals,tp3_vals)

MRI_image_types=["ADC","MPE","TTP"]
for image_type in MRI_image_types:
    tp1_vals,tp2_vals,tp3_vals=getIQRoverMedian(image_type,timepoints,MRI_functional_data)
    print(image_type)
    print(tp1_vals)
    print(tp2_vals)
    print(tp3_vals)