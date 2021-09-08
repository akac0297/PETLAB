#!/usr/bin/env python3
# coding: utf-8

from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ADC_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_ADC_analysis_B50T.csv")
MPE_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_MPE_analysis_new.csv")
TTP_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_TTP_analysis_new.csv")
SUV_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_new.csv")

timepoints=[1.0,2.0,3.0]

def makeHistograms(feature,timepoints,functional_data,image_type):
    subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
    tp1=subset1[feature].to_list()

    subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
    tp2=subset2[feature].to_list()

    subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
    tp3=subset3[feature].to_list()

    plt.figure()
    plt.hist(tp1,bins=50,color='g',label='Time point 1')
    plt.hist(tp2,bins=50,color='b', label= 'Time point 2')
    plt.hist(tp3,bins=50,color='r', label='Time point 3')
    plt.legend()
    plt.title(f"{feature} histogram for {image_type}")
    plt.savefig(f"/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/Histograms/{feature}_histogram_for_{image_type}.png")

features = ["95% ADC","MEDIAN ADC","MEAN ADC","STD DEV ADC","IQR ADC", "MEDIAN ABS DEV ADC", "5% ADC"]
for feature in features:
    makeHistograms(feature,timepoints,ADC_data,"ADC")

features = ["95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","MEDIAN ABS DEV MPE","5% MPE"]
for feature in features:
    makeHistograms(feature,timepoints,MPE_data,"MPE")

features = ["95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","MEDIAN ABS DEV TTP","5% TTP"]
for feature in features:
    makeHistograms(feature,timepoints,TTP_data,"TTP")

features = ["95% SUV","MAX SUV","MEDIAN SUV","MEAN SUV","STD DEV SUV","IQR SUV","MEDIAN ABS DEV SUV","5% SUV"]
for feature in features:
    makeHistograms(feature,timepoints,SUV_data,"SUV")