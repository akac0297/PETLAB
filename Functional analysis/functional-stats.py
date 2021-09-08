#!/usr/bin/env python3
# coding: utf-8

from scipy import stats
import pandas as pd
import scikit_posthocs as sp

"""
Kruskal-Wallis test - functional analysis
"""

MRI_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_functional_analysis_new.csv")
ADC_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_ADC_analysis_B50T.csv")
timepoints=[1.0,2.0,3.0]

def runKW(feature,timepoints,functional_data):
    subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
    tp1=subset1[feature].to_list()

    subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
    tp2=subset2[feature].to_list()

    subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
    tp3=subset3[feature].to_list()
    _, pvalue = stats.kruskal(tp1,tp2,tp3)
    if pvalue<0.05:
        print(feature, stats.kruskal(tp1,tp2,tp3))

def runDunn(feature,timepoints,functional_data):
    subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
    tp1=subset1[feature].to_list()

    subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
    tp2=subset2[feature].to_list()

    subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
    tp3=subset3[feature].to_list()

    data=[tp1, tp2, tp3]
    result=sp.posthoc_dunn(data,p_adjust='bonferroni')
    #result.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/Dunn dataframes/{feature}_dataframe_Dunn.csv")
    return(result)

features = ["95% ADC","MEDIAN ADC","MEAN ADC","STD DEV ADC","IQR ADC", "MEDIAN ABS DEV ADC", "5% ADC"]
for feature in features:
    runKW(feature,timepoints,ADC_data)
    result=runDunn(feature,timepoints,ADC_data)
    print(feature)
    print(result)

features = ["95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","MEDIAN ABS DEV MPE","5% MPE"]
for feature in features:
    runKW(feature,timepoints,MRI_functional_data)
    result=runDunn(feature,timepoints,MRI_functional_data)
    print(feature)
    print(result)

features = ["95% TTP","MEDIAN TTP","MEAN TTP","STD DEV TTP","IQR TTP","MEDIAN ABS DEV TTP","5% TTP"]
for feature in features:
    runKW(feature,timepoints,MRI_functional_data)
    result=runDunn(feature,timepoints,MRI_functional_data)
    print(feature)
    print(result)

SUV_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_new.csv")
SUV_functional_data=SUV_functional_data.drop([39,40,41])
features = ["95% SUV","MAX SUV","MEDIAN SUV","MEAN SUV","STD DEV SUV","IQR SUV","MEDIAN ABS DEV SUV","5% SUV"]
for feature in features:
    runKW(feature,timepoints,SUV_functional_data)
    result=runDunn(feature,timepoints,SUV_functional_data)
    print(feature)
    print(result)
