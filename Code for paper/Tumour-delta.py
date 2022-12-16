#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

MRI_patient_list=("06","14","16","19","21","23")
binCounts=("32","64","128","256","None")

def getDataframes(patient_id,binCount):
    df=pd.read_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_tumour_WES_0{patient_id}_binCount{binCount}_df.csv")

    # DCE_df = df.loc[df['modality'] == ('DCE ME')]
    # T1w_df=df.loc[df['modality'] == ('T1w')]
    # T1w_z_df=T1w_df.loc[T1w_df['Z-score normalisation'] == "True"]
    # T1w_df=T1w_df.loc[T1w_df['Z-score normalisation'] == "False"]
    # T2w_df=df.loc[df['modality'] == ('T2w SPAIR')]
    # T2w_z_df=T2w_df.loc[T2w_df['Z-score normalisation'] == "True"]
    # T2w_df=T2w_df.loc[T2w_df['Z-score normalisation'] == "False"]
    # ADC_df=df.loc[df['modality'] == ('ADC')]

    Bef_df=df[df['time label'].str.contains("Before")]
    Dur_df=df[df['time label'].str.contains("During")]
    Post_df=df[df['time label'].str.contains("Post ")]

    patients= Bef_df["Patient"].reset_index(drop=True)
    modalities = Bef_df["modality"].reset_index(drop=True)
    normalisation = Bef_df["Z-score normalisation"].reset_index(drop=True)

    Bef_df=Bef_df.drop(['Patient', 'time label','modality','Z-score normalisation','Unnamed: 0'], axis=1)
    Dur_df=Dur_df.drop(['Patient', 'time label','modality','Z-score normalisation','Unnamed: 0'], axis=1)
    Post_df=Post_df.drop(['Patient', 'time label','modality','Z-score normalisation','Unnamed: 0'], axis=1)

    delta_df_12=(Bef_df.sub(Dur_df.values.astype(np.float64))).div(Bef_df.values.astype(np.float64))
    delta_df_13=(Bef_df.sub(Post_df.values.astype(np.float64))).div(Bef_df.values.astype(np.float64))
    delta_df_23=(Dur_df.sub(Post_df.values.astype(np.float64))).div(Bef_df.values.astype(np.float64))

    delta_df_12=delta_df_12.reset_index(drop=True)
    delta_df_12["Patient"]=patients
    delta_df_12["modality"]=modalities
    delta_df_12["Z-score normalisation"]=normalisation

    delta_df_13=delta_df_13.reset_index(drop=True)
    delta_df_13["Patient"]=patients
    delta_df_13["modality"]=modalities
    delta_df_13["Z-score normalisation"]=normalisation

    delta_df_23=delta_df_23.reset_index(drop=True)
    delta_df_23["Patient"]=patients
    delta_df_23["modality"]=modalities
    delta_df_23["Z-score normalisation"]=normalisation

    # print(delta_df_12)
    # print(delta_df_13)
    # print(delta_df_23)

    # delta_df_12.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_WES_0{patient_id}_binCount{binCount}_12.csv")
    # delta_df_13.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_WES_0{patient_id}_binCount{binCount}_13.csv")
    # delta_df_23.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_WES_0{patient_id}_binCount{binCount}_23.csv")

    return delta_df_12,delta_df_13,delta_df_23

for binCount in binCounts:
    df_12 = pd.DataFrame()
    df_13 = pd.DataFrame()
    df_23 = pd.DataFrame()
    for patient_id in MRI_patient_list:
        delta_df_12,delta_df_13,delta_df_23 = getDataframes(patient_id,binCount)
        df_12=pd.concat([df_12,delta_df_12], axis=0, ignore_index=True)
        df_13=pd.concat([df_13,delta_df_13], axis=0, ignore_index=True)
        df_23=pd.concat([df_23,delta_df_23], axis=0, ignore_index=True)
    df_12.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_binCount{binCount}_12.csv")
    df_13.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_binCount{binCount}_13.csv")
    df_23.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Delta_radiomics_tumour_binCount{binCount}_23.csv")
    print(f"Bin count {binCount} delta radiomics calculation complete")