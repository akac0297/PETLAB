#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from scipy.stats import pearsonr, spearmanr

"""
Obtain shortened dataframes from DCE-MRI and PET radiomics (feature selection using features obtained from the literature)
"""
df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features.csv")
DCE_df = df[['Idmn', 'Idn','SumAverage','JointAverage','Minimum','Entropy','ClusterTendency','JointEnergy','DifferenceVariance','SumEntropy','image label','Patient']]
DCE_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features_shortened.csv")

df1=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_radiomics_features.csv")
PET_df = df1[['ClusterProminence','Mean','Id','Idm','InverseVariance','GrayLevelNonUniformity','SmallAreaHighGrayLevelEmphasis','LargeAreaHighGrayLevelEmphasis','image label','Patient']]
#PET_df=PET_df[PET_df['Patient']!=(9 or 13)]
PET_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_radiomics_features_shortened_new.csv")

def obtainXy(modality='DCE'):
    if modality=='DCE':
        DCE_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features_shortened.csv")
        DCE_df=DCE_df.drop(['Patient', 'image label','Unnamed: 0'], axis=1)
        columns=list(DCE_df)
        image_labels=['Bef Pre','Bef Post1', 'Bef Post2', 'Bef Post3', 'Bef Post4', 'Bef Post 5','Dur Pre','Dur Post1', 'Dur Post2', 'Dur Post3', 'Dur Post4', 'Dur Post 5']

        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)

        data=DCE_df.values
        X = data.reshape(-1,120) #each individual DCE image produces 10 features, with 12 images per patient, so 120 total radiomics features per patient
        y = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]).reshape(-1,1)

        X_df=pd.DataFrame(X,columns=[new_labels])
    elif modality=='PET':
        PET_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_radiomics_features_shortened_new.csv")
        #PET_df=PET_df[PET_df['Patient']!=(9 and 13)]
        PET_df=PET_df.drop(['Patient', 'image label','Unnamed: 0'], axis=1)
        columns=list(PET_df)
        image_labels=['Bef PET','Dur PET']
        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)

        data=PET_df.values
        X = data.reshape(-1,16) #each individual PET image produces 8 features, with 2 images per patient, so 16 total radiomics features per patient
        y = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]).reshape(-1,1)
        #y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]).reshape(-1,1)
        print(X.shape)
        print(y.shape)
        X_df=pd.DataFrame(X,columns=[new_labels])
    return(X,y,X_df)

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

def calculate_SM_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues

def getXRelevant(X,X_df,y):
    relevant_features = mrmr_classif(X,y,K=10)
    print(relevant_features)
    X_relevant = X[:,relevant_features]
    X_relevant_df=X_df.iloc[:,relevant_features]
    X_relevant_df['Tumour volume']=y
    return X_relevant_df,X_relevant

def calculateCorrelation(X_relevant_df,modality):
    corr_df=X_relevant_df.corr()
    print(corr_df)

    p_val_df=calculate_pvalues(corr_df)
    print(p_val_df)
    corr_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Pearson_corr_df_{modality}_radiomics.csv')
    p_val_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Pearson_p_val_df_{modality}_radiomics.csv')

def calculateSMCorrelation(X_relevant_df,modality):
    corr_df=X_relevant_df.corr(method='spearman')
    print(corr_df)

    p_val_df=calculate_SM_pvalues(corr_df)
    print(p_val_df)
    corr_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Spearman_corr_df_{modality}_radiomics.csv')
    p_val_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Spearman_p_val_df_{modality}_radiomics.csv')

"""
May want to run MRMR to reduce features of X first.
Then want to get a correlation matrix

"""

"""
DCE-MRI radiomics features
- Idmn
- Idn
- SumAverage
- JointAverage
- Minimum
- Entropy
- ClusterTendency
- JointEnergy
- DifferenceVariance
- SumEntropy


PET radiomics features
- ClusterProminence
- InverseVariance
- Id
- Idm
- Mean
- NGLDM number nonuniformity ** don't have this one. Can try GrayLevelNonUniformity
- HISZE ** don't have this one but can try SmallAreaHighGrayLevelEmphasis
- HIZE ** don't have this one but can try LargeAreaHighGrayLevelEmphasis

"""


"""
Determine whether tumour volume changed in T2w imaging and PET imaging

T2w:
- read in the volume information (TP 1, 2, 3)
- subtract TP 3 volume from TP 1 volume then divide by TP 1 volume and store as vector
- find median percentage volume change - set to 1 if the value is above the median and 0 if it is below

"""

vol_df=pd.read_csv('/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volume_analysis_new.csv')
PET_df=vol_df[vol_df['IMAGE_TYPE']=='PET']
MRI_df=vol_df[vol_df['IMAGE_TYPE']=='T2w MRI']

PET_tp1=PET_df[PET_df['TIMEPOINT']==1]['TUMOUR VOLUME_CM3']
PET_tp3=PET_df[PET_df['TIMEPOINT']==3]['TUMOUR VOLUME_CM3']
PET_tp1_data=PET_tp1.values
PET_tp3_data=PET_tp3.values
PET_pc_data=(PET_tp1_data-PET_tp3_data)/PET_tp1_data*100

PET_pc_data=PET_pc_data[:-3]

median_PET=np.median(PET_pc_data)
#print(median_PET)
PET_result=[0 if i < median_PET else 1 for i in PET_pc_data]
#print(PET_result)

# output: tumour volume results for PET: [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1] (patients 4 to 18)

MRI_tp1=MRI_df[MRI_df['TIMEPOINT']==1]['TUMOUR VOLUME_CM3']
MRI_tp3=MRI_df[MRI_df['TIMEPOINT']==3]['TUMOUR VOLUME_CM3']
MRI_tp1_data=MRI_tp1.values
MRI_tp3_data=MRI_tp3.values
MRI_pc_data=(MRI_tp1_data-MRI_tp3_data)/MRI_tp1_data*100

median_MRI=np.median(MRI_pc_data)
#print(median_MRI)

MRI_result=[0 if i < median_MRI else 1 for i in MRI_pc_data]
#print(MRI_result)

# output: tumour volume results for T2w MRI: [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0] (patients 4 to 23)