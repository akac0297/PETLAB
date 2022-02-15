#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from scipy.stats import pearsonr, spearmanr

"""
Feature selection for DCE-MRI and PET radiomics (using features obtained from the literature). Outputs reduced dataframes of relevant features

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

ADC (DWI-MRI) features
- mean
- difference value (maximum minus minimum)
- joint energy
- joint entropy
- entropy
- busyness
- LongRunLowGrayLevelEmphasis

T2w-MRI radiomics features
- entropy
- uniformity
- busyness
- Imc2
- LongRunLowGrayLevelEmphasis
- Kurtosis

PET radiomics features
- ClusterProminence
- Mean
- Id
- Idm
- InverseVariance
- NGLDM number nonuniformity ** Use Size-Zone Non-Uniformity
- HISZE ** Use SmallAreaHighGrayLevelEmphasis
- HIZE ** Use LargeAreaHighGrayLevelEmphasis

"""
df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_Feb_14_22.csv")
DCE_df = df.loc[df['modality'] == ('DCE')]
DCE_df = DCE_df[['Idmn', 'Idn','SumAverage','JointAverage','Minimum','Entropy','ClusterTendency','JointEnergy','DifferenceVariance','SumEntropy','time label','modality','Patient']]
DCE_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features_shortened_Feb_14_22.csv")

ADC_df = df.loc[df['modality'] == ('ADC')]
ADC_df = ADC_df[['Minimum','Mean','Maximum','Entropy','JointEnergy','JointEntropy','Busyness','LongRunLowGrayLevelEmphasis','time label','modality','Patient']]
ADC_df['DifferenceValue']=ADC_df['Maximum']-ADC_df['Minimum']
ADC_df=ADC_df.drop(['Minimum','Maximum'],axis=1)
ADC_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC-MRI_radiomics_features_shortened_Feb_14_22.csv")

T2w_df=df.loc[df['modality'] == ('T2w SPAIR')]
T2w_df=T2w_df[['Entropy','Uniformity','Busyness','Imc2','LongRunLowGrayLevelEmphasis','Kurtosis','time label','modality','Patient']]
T2w_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2w-MRI_radiomics_features_shortened_Feb_14_22.csv")

PET_df = df.loc[df['modality'] == ('PET')]
PET_df = PET_df[['ClusterProminence','Mean','Id','Idm','InverseVariance','SizeZoneNonUniformity','SmallAreaHighGrayLevelEmphasis','LargeAreaHighGrayLevelEmphasis','time label','modality','Patient']]
#PET_df=PET_df[PET_df['Patient']!=(9 or 13)]
PET_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_radiomics_features_shortened_Feb_14_22.csv")

def obtainX(df,modality):
    df=df.drop(['Patient', 'time label','modality'], axis=1) #,'Unnamed: 0'
    columns=list(df)
    data=df.to_numpy()

    if modality == "DCE":
        image_labels=['Bef Pre','Bef Post1', 'Bef Post2', 'Bef Post3', 'Bef Post4', 'Bef Post 5','Bef ME','Dur Pre','Dur Post1', 'Dur Post2', 'Dur Post3', 'Dur Post4', 'Dur Post 5', 'Dur ME','Aft Pre','Aft Post1', 'Aft Post2', 'Aft Post3', 'Aft Post4', 'Aft Post 5', 'Aft ME']

        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)
        
        X = data.reshape(-1,210) #each individual DCE image produces 10 features, with 21 images per patient, so 210 total radiomics features per patient
        #X=X[0]
        X_df=pd.DataFrame(X,columns=[new_labels])
    
    elif modality == "ADC":
        image_labels=['Bef ADC','Dur ADC', 'Aft ADC']
        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)

        X = data.reshape(-1,21) #each individual ADC image produces 7 features, with 3 images per patient, so 21 total radiomics features per patient
        #X=X[0]
        X_df=pd.DataFrame(X,columns=[new_labels])

    elif modality == "T2w SPAIR":
        image_labels=['Bef T2w SPAIR','Dur T2w SPAIR', 'Aft T2w SPAIR']
        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)

        X = data.reshape(-1,18) #each individual T2w SPAIR image produces 6 features, with 3 images per patient, so 18 total radiomics features per patient
        #X=X[0]
        X_df=pd.DataFrame(X,columns=[new_labels])

    elif modality == "PET":
        image_labels=['Bef PET','Dur PET', 'Aft PET']
        new_labels=[]
        for feature in columns:
            for label in image_labels:
                new_labels.append(feature + " " + label)

        X = data.reshape(-1,24) #each individual PET image produces 8 features, with 3 images per patient, so 24 total radiomics features per patient
        #X=X[0]
        X_df=pd.DataFrame(X,columns=[new_labels])

    return(X,X_df)

dce,dce_df=obtainX(DCE_df,'DCE')
print(dce)
print(dce_df)
adc,adc_df=obtainX(ADC_df,'ADC')
t2w,t2w_df=obtainX(T2w_df,'T2w SPAIR')
pet,pet_df=obtainX(PET_df,'PET')

DCE_ADC_df=pd.concat([dce_df,adc_df],axis=1)
DCE_ADC=DCE_ADC_df.to_numpy()[0]
DCE_T2W_df=pd.concat([dce_df,t2w_df],axis=1)
DCE_T2W=DCE_T2W_df.to_numpy()[0]
DCE_PET_df=pd.concat([dce_df,pet_df],axis=1)
DCE_PET=DCE_PET_df.to_numpy()[0]
ADC_T2W_df=pd.concat([adc_df,t2w_df],axis=1)
ADC_T2W=ADC_T2W_df.to_numpy()[0]
ADC_PET_df=pd.concat([adc_df,pet_df],axis=1)
ADC_PET=ADC_PET_df.to_numpy()[0]
T2W_PET_df=pd.concat([t2w_df,pet_df],axis=1)
T2W_PET=T2W_PET_df.to_numpy()[0]
DCE_ADC_T2W_df=pd.concat([dce_df,adc_df,t2w_df],axis=1)
DCE_ADC_T2W=DCE_ADC_T2W_df.to_numpy()[0]
DCE_ADC_PET_df=pd.concat([dce_df,adc_df,pet_df],axis=1)
DCE_ADC_PET=DCE_ADC_PET_df.to_numpy()[0]
ADC_T2W_PET_df=pd.concat([adc_df,t2w_df,pet_df],axis=1)
ADC_T2W_PET=ADC_T2W_PET_df.to_numpy()[0]
DCE_ADC_T2W_PET_df=pd.concat([dce_df,adc_df,t2w_df,pet_df],axis=1)
DCE_ADC_T2W_PET=DCE_ADC_T2W_PET_df.to_numpy()[0]

#print(DCE_PET)
#print(DCE_PET_df)

def getXRelevant(X,X_df,y):
    relevant_features = mrmr_classif(X,y,K=10)
    print(relevant_features)
    X_relevant = X[:,relevant_features]
    X_relevant_df=X_df.iloc[:,relevant_features]
    return X_relevant_df,X_relevant # After this need to save the dataframe as csv with reduced/selected radiomics features for each modality

# example!
y=[0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
y=[0]
# e.g. output: tumour volume results for PET: y = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1] (patients 4 to 18)

dce_relevant_df,dce_relevant=getXRelevant(dce,dce_df,y)
adc_relevant_df,adc_relevant=getXRelevant(adc,adc_df,y)
t2w_relevant_df,t2w_relevant=getXRelevant(t2w,t2w_df,y)
pet_relevant_df,pet_relevant=getXRelevant(pet,dce_df,y)
dce_adc_relevant_df,dce_adc_relevant=getXRelevant(DCE_ADC,DCE_ADC_df,y)
dce_t2w_relevant_df,dce_t2w_relevant=getXRelevant(DCE_T2W,DCE_T2W_df,y)
dce_pet_relevant_df,dce_pet_relevant=getXRelevant(DCE_PET,DCE_PET_df,y)
adc_t2w_relevant_df,adc_t2w_relevant=getXRelevant(ADC_T2W,ADC_T2W_df,y)
adc_pet_relevant_df,adc_pet_relevant=getXRelevant(ADC_PET,ADC_PET_df,y)
t2w_pet_relevant_df,t2w_pet_relevant=getXRelevant(T2W_PET,T2W_PET_df,y)
dce_adc_t2w_relevant_df,dce_adc_t2w_relevant=getXRelevant(DCE_ADC_T2W,DCE_ADC_T2W_df,y)
dce_adc_pet_relevant_df,dce_adc_pet_relevant=getXRelevant(DCE_ADC_PET,DCE_ADC_PET_df,y)
adc_t2w_pet_relevant_df,adc_t2w_pet_relevant=getXRelevant(ADC_T2W_PET,ADC_T2W_PET_df,y)
dce_adc_t2w_pet_relevant_df,dce_adc_t2w_pet_relevant=getXRelevant(DCE_ADC_T2W_PET,DCE_ADC_T2W_PET_df,y)

dce_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df_Feb_14_22.csv")
adc_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_relevant_df_Feb_14_22.csv")
t2w_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2W_relevant_df_Feb_14_22.csv")
pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_relevant_df_Feb_14_22.csv")
dce_adc_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_relevant_df_Feb_14_22.csv")
dce_t2w_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_T2W_relevant_df_Feb_14_22.csv")
dce_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_PET_relevant_df_Feb_14_22.csv")
adc_t2w_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_T2W_relevant_df_Feb_14_22.csv")
adc_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_PET_relevant_df_Feb_14_22.csv")
t2w_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2W_PET_relevant_df_Feb_14_22.csv")
dce_adc_t2w_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_T2W_relevant_df_Feb_14_22.csv")
dce_adc_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_PET_relevant_df_Feb_14_22.csv")
adc_t2w_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_T2W_PET_relevant_df_Feb_14_22.csv")
dce_adc_t2w_pet_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_T2W_PET_relevant_df_Feb_14_22.csv")

"""
Run MRMR to reduce features of X first.
Then get a correlation matrix
"""
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

def calculateCorrelation(X_relevant_df,modality):
    corr_df=X_relevant_df.corr()
    print(corr_df)

    p_val_df=calculate_pvalues(corr_df)
    print(p_val_df)
    corr_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Pearson_corr_df_{modality}_radiomics_14_Feb_22.csv')
    p_val_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Pearson_p_val_df_{modality}_radiomics_14_Feb_22.csv')

def calculateSMCorrelation(X_relevant_df,modality):
    corr_df=X_relevant_df.corr(method='spearman')
    print(corr_df)

    p_val_df=calculate_SM_pvalues(corr_df)
    print(p_val_df)
    corr_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Spearman_corr_df_{modality}_radiomics_14_Feb_22.csv')
    p_val_df.to_csv(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Spearman_p_val_df_{modality}_radiomics_14_Feb_22.csv')

calculateCorrelation(dce_relevant_df,"dce")
calculateSMCorrelation(dce_relevant_df,"dce")
calculateCorrelation(adc_relevant_df,"adc")
calculateSMCorrelation(adc_relevant_df,"adc")
calculateCorrelation(t2w_relevant_df,"t2w")
calculateSMCorrelation(t2w_relevant_df,"t2w")
calculateCorrelation(pet_relevant_df,"pet")
calculateSMCorrelation(pet_relevant_df,"pet")
calculateCorrelation(dce_adc_relevant_df,"dce_adc")
calculateSMCorrelation(dce_adc_relevant_df,"dce_adc")
calculateCorrelation(dce_t2w_relevant_df,"dce_t2w")
calculateSMCorrelation(dce_t2w_relevant_df,"dce_t2w")
calculateCorrelation(dce_pet_relevant_df,"dce_pet")
calculateSMCorrelation(dce_pet_relevant_df,"dce_pet")
calculateCorrelation(adc_t2w_relevant_df,"adc_t2w")
calculateSMCorrelation(adc_t2w_relevant_df,"adc_t2w")
calculateCorrelation(adc_pet_relevant_df,"adc_pet")
calculateSMCorrelation(adc_pet_relevant_df,"adc_pet")
calculateCorrelation(t2w_pet_relevant_df,"t2w_pet")
calculateSMCorrelation(t2w_pet_relevant_df,"t2w_pet")
calculateCorrelation(dce_adc_t2w_relevant_df,"dce_adc_t2w")
calculateSMCorrelation(dce_adc_t2w_relevant_df,"dce_adc_t2w")
calculateCorrelation(dce_adc_pet_relevant_df,"dce_adc_pet")
calculateSMCorrelation(dce_adc_pet_relevant_df,"dce_adc_pet")
calculateCorrelation(adc_t2w_pet_relevant_df,"adc_t2w_pet")
calculateSMCorrelation(adc_t2w_pet_relevant_df,"adc_t2w_pet")
calculateCorrelation(dce_adc_t2w_pet_relevant_df,"dce_adc_t2w_pet")
calculateSMCorrelation(dce_adc_t2w_pet_relevant_df,"dce_adc_t2w_pet")

"""
Determine whether tumour volume changed in T2w imaging and PET imaging
"""
# input y (response classifications) from Volume-analysis.py

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