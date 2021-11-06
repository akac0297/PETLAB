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

def runWelchVol(image_type,timepoints,volume_data):
    subset1=volume_data[volume_data["TIMEPOINT"]==timepoints[0]]
    subset1=subset1[subset1["IMAGE_TYPE"]==image_type]
    tp1=subset1["TUMOUR VOLUME_CM3"].to_list()

    subset2=volume_data[volume_data["TIMEPOINT"]==timepoints[1]]
    subset2=subset2[subset2["IMAGE_TYPE"]==image_type]
    tp2=subset2["TUMOUR VOLUME_CM3"].to_list()

    subset3=volume_data[volume_data["TIMEPOINT"]==timepoints[2]]
    subset3=subset3[subset3["IMAGE_TYPE"]==image_type]
    tp3=subset3["TUMOUR VOLUME_CM3"].to_list()

    #perform Welch's t-test
    _, p_value_12 = stats.ttest_ind(tp1, tp2, equal_var = False)
    _, p_value_13 = stats.ttest_ind(tp1, tp3, equal_var = False)
    _, p_value_23 = stats.ttest_ind(tp2, tp3, equal_var = False)
    print("p-value for time points 1 to 2", p_value_12)
    print("p-value for time points 1 to 3", p_value_13)
    print("p-value for time points 2 to 3", p_value_23)

image_types = ["MPE MRI", "T2w MRI", "B50T MRI", "B800T MRI"]
for image_type in image_types:
    print(image_type)
    runWelchVol(image_type,timepoints,volume_data)

PET_data=volume_data[volume_data["IMAGE_TYPE"]=="PET"]
PET_data=PET_data.drop([195,200,205,210,215,220,225,230,235])
print("PET")
runWelchVol("PET",timepoints,PET_data)

"""
Welch independent samples t-test - functional analysis
"""

MRI_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_functional_analysis_22-Sep-v1.csv")
ADC_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/MRI_ADC_analysis_B800T.csv")

timepoints=[1,2,3]

def returnSubsets(feature,timepoints,functional_data):
    subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
    tp1=subset1[feature].to_list()

    subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
    tp2=subset2[feature].to_list()

    subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
    tp3=subset3[feature].to_list()
    
    return(tp1,tp2,tp3)

def runWelch(tp1,tp2,tp3):
    #perform Welch's t-test
    _, p_value_12 = stats.ttest_ind(tp1, tp2, equal_var = False)
    _, p_value_13 = stats.ttest_ind(tp1, tp3, equal_var = False)
    _, p_value_23 = stats.ttest_ind(tp2, tp3, equal_var = False)

    return(p_value_12,p_value_13,p_value_23)

features = ["95% ADC","MEDIAN ADC","MEAN ADC","STD DEV ADC","IQR ADC", "MEDIAN ABS DEV ADC", "5% ADC"]
for feature in features:
    tp1,tp2,tp3 = returnSubsets(feature,timepoints,ADC_data)
    p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
    print(feature)
    print("P-value TP 1 to 2:", p_value_12)
    print("P-value TP 1 to 3:", p_value_13)
    print("P-value TP 2 to 3:", p_value_23)

features = ["95% MPE","MEDIAN MPE","MEAN MPE","STD DEV MPE","IQR MPE","MEDIAN ABS DEV MPE","5% MPE"]
for feature in features:
    tp1,tp2,tp3 = returnSubsets(feature,timepoints,MRI_functional_data)
    p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
    print(feature)
    print("P-value TP 1 to 2:", p_value_12)
    print("P-value TP 1 to 3:", p_value_13)
    print("P-value TP 2 to 3:", p_value_23)

SUV_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_updated.csv")
SUV_functional_data=SUV_functional_data.drop([39,40,41])
features = ["95% SUV","MAX SUV", "MIN SUV", "MEDIAN SUV","MEAN SUV","STD DEV SUV","IQR SUV","MEDIAN ABS DEV SUV","5% SUV"]
for feature in features:
    tp1,tp2,tp3 = returnSubsets(feature,timepoints,SUV_functional_data)
    p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
    print(feature)
    print("P-value TP 1 to 2:", p_value_12)
    print("P-value TP 1 to 3:", p_value_13)
    print("P-value TP 2 to 3:", p_value_23)

#Heterogeneity
features = ["MEDIAN ADC","IQR ADC"]
median=features[0]
tp1_m,tp2_m,tp3_m = returnSubsets(median,timepoints,ADC_data)
iqr=features[1]
tp1_i,tp2_i,tp3_i = returnSubsets(iqr,timepoints,ADC_data)

tp1 = [i / j for i, j in zip(tp1_i, tp1_m)]
tp2 = [i / j for i, j in zip(tp2_i, tp2_m)]
tp3 = [i / j for i, j in zip(tp3_i, tp3_m)]

p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
print("ADC heterogeneity")
print("P-value TP 1 to 2:", p_value_12)
print("P-value TP 1 to 3:", p_value_13)
print("P-value TP 2 to 3:", p_value_23)

features = ["MEDIAN MPE","IQR MPE"]
median=features[0]
tp1_m,tp2_m,tp3_m = returnSubsets(median,timepoints,MRI_functional_data)
iqr=features[1]
tp1_i,tp2_i,tp3_i = returnSubsets(iqr,timepoints,MRI_functional_data)

tp1 = [i / j for i, j in zip(tp1_i, tp1_m)]
tp2 = [i / j for i, j in zip(tp2_i, tp2_m)]
tp3 = [i / j for i, j in zip(tp3_i, tp3_m)]

p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
print("MPE heterogeneity")
print("P-value TP 1 to 2:", p_value_12)
print("P-value TP 1 to 3:", p_value_13)
print("P-value TP 2 to 3:", p_value_23)

SUV_functional_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/PET_functional_analysis_updated.csv")
SUV_functional_data=SUV_functional_data.drop([39,40,41])
features = ["MEDIAN SUV","IQR SUV"]
median=features[0]
tp1_m,tp2_m,tp3_m = returnSubsets(median,timepoints,SUV_functional_data)
iqr=features[1]
tp1_i,tp2_i,tp3_i = returnSubsets(iqr,timepoints,SUV_functional_data)

tp1 = [i / j for i, j in zip(tp1_i, tp1_m)]
tp2 = [i / j for i, j in zip(tp2_i, tp2_m)]
tp3 = [i / j for i, j in zip(tp3_i, tp3_m)]

p_value_12,p_value_13,p_value_23=runWelch(tp1,tp2,tp3)
print("SUV heterogeneity")
print("P-value TP 1 to 2:", p_value_12)
print("P-value TP 1 to 3:", p_value_13)
print("P-value TP 2 to 3:", p_value_23)
