#!/usr/bin/env python
# coding: utf-8

from scipy import stats
import pandas as pd
import scikit_posthocs as sp

"""
Welch independent samples t-test - volume analysis
"""

volume_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volume_analysis_new.csv")

timepoints=[1,2,3]

def returnSubsets(image_type,timepoints,volume_data):
    subset1=volume_data[volume_data["TIMEPOINT"]==timepoints[0]]
    subset1=subset1[subset1["IMAGE_TYPE"]==image_type]
    tp1=subset1["TUMOUR VOLUME_CM3"].to_list()

    subset2=volume_data[volume_data["TIMEPOINT"]==timepoints[1]]
    subset2=subset2[subset2["IMAGE_TYPE"]==image_type]
    tp2=subset2["TUMOUR VOLUME_CM3"].to_list()

    subset3=volume_data[volume_data["TIMEPOINT"]==timepoints[2]]
    subset3=subset3[subset3["IMAGE_TYPE"]==image_type]
    tp3=subset3["TUMOUR VOLUME_CM3"].to_list()
    #_, pvalue = stats.kruskal(tp1,tp2,tp3)
    #if pvalue!=1:
    #    print(image_type, stats.kruskal(tp1,tp2,tp3))
    
    return(tp1,tp2,tp3)

def runWelch(tp1,tp2,tp3):
    #perform Welch's t-test
    _, p_value_12 = stats.ttest_ind(tp1, tp2, equal_var = False)
    _, p_value_13 = stats.ttest_ind(tp1, tp3, equal_var = False)
    _, p_value_23 = stats.ttest_ind(tp2, tp3, equal_var = False)

    return(p_value_12,p_value_13,p_value_23)

image_types = ["MPE MRI", "T2w MRI", "B50T MRI", "B800T MRI"]
for image_type in image_types:
    tp1,tp2,tp3 = returnSubsets(image_type,timepoints,volume_data)
    p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
    print(image_type)
    print("P-value TP 1 to 2:", p_value_12)
    print("P-value TP 1 to 3:", p_value_13)
    print("P-value TP 2 to 3:", p_value_23)

PET_data=volume_data[volume_data["IMAGE_TYPE"]=="PET"]
PET_data=PET_data.drop([195,200,205,210,215,220,225,230,235])
tp1,tp2,tp3 = returnSubsets(image_type,timepoints,volume_data)
p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
print("PET")
print("P-value TP 1 to 2:", p_value_12)
print("P-value TP 1 to 3:", p_value_13)
print("P-value TP 2 to 3:", p_value_23)