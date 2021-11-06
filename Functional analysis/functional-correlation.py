#!/usr/bin/env python
# coding: utf-8

import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import stats
import os

"""
Code to generate matrices of pearson correlation coefficients between the different functional parameters

input is functional analysis (MRI and PET) dataframes
separate out into time points and feature (median: ADC, MPE, TTP, SUV) for patients 4-19

calculate pearson correlation coefficients between the median features across all these patients

we want to output 3 spreadsheets (one for each time point)

"""
timepoints=[1,2,3]

functional_data=pd.read_csv("/home/alicja/Downloads/Updated functional data - MRI_functional_analysis_new2.csv")
functional_data=functional_data.drop([39,40,41,42,43,44,45,46])
#print(functional_data)

functional_data=functional_data[["TIMEPOINT","MEDIAN ADC","MEDIAN MPE", "MEDIAN TTP", "MEDIAN SUV"]]

subset1=functional_data[functional_data["TIMEPOINT"]==timepoints[0]]
subset1=subset1.drop(["TIMEPOINT"],axis=1)
subset2=functional_data[functional_data["TIMEPOINT"]==timepoints[1]]
subset2=subset2.drop(["TIMEPOINT"],axis=1)
subset3=functional_data[functional_data["TIMEPOINT"]==timepoints[2]]
subset3=subset3.drop(["TIMEPOINT"],axis=1)

#print(subset1)
print("Time point 1")
print(subset1.corr(method="pearson"))
corr1=subset1.corr(method="pearson")
print("Time point 2")
print(subset2.corr(method="pearson"))
corr2=subset2.corr(method="pearson")
print("Time point 3")
#print(subset3.corr(method="pearson"))
corr3=subset3.corr(method="pearson")
print(corr3)

corr1.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/Corr-tp-1-functional-analysis.csv")
corr2.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/Corr-tp-2-functional-analysis.csv")
corr3.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Functional analysis/Corr-tp-3-functional-analysis.csv")