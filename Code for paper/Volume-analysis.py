#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]
path="/home/alicja/PET_LAB_PROCESSED/"

MRI_seg_path="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_MRI_tumour_masks/"
PET_seg_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/New PET tumours/"

def getPETImg(patient_no,timepoint,PET_seg_path):
    PET_img=PET_seg_path+f"WES_0{patient_no}_TIMEPOINT_{timepoint}_PET_TUMOUR.nii.gz"
    return(PET_img)

def getMRIImg(patient_no,timepoint,MRI_seg_path):
    MRI_img=MRI_seg_path+f"tumour_mask_WES_0{patient_no}_TIMEPOINT_{timepoint}_DCE_ACQ_1.nii.gz"
    return(MRI_img)

def getPETVolume(patient_no,timepoint,PET_seg_path):
    PET_img=getPETImg(patient_no,timepoint,PET_seg_path)
    seg=sitk.ReadImage(PET_img)
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array>0)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])/1000
    return(volume)

def getMRIVolume(patient_no,timepoint,MRI_seg_path):
    MRI_img=getMRIImg(patient_no,timepoint,MRI_seg_path)
    seg=sitk.ReadImage(MRI_img)
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array==1)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])/1000
    return(volume)

def getVolDF(patient_list,timepoints,PET_seg_path,MRI_seg_path):
    df=pd.DataFrame(columns=["PATIENT_ID","IMAGE_TYPE","TIMEPOINT","TUMOUR VOLUME_CM3"])
    for patient_no in patient_list:
        offset=(patient_list.index(patient_no))*6
        for timepoint in timepoints:
            PET_img=getPETImg(patient_no,timepoint,PET_seg_path)
            if os.path.isfile(PET_img):
                PET_vol=getPETVolume(patient_no,timepoint,PET_seg_path)
            else:
                PET_vol=np.NaN
            tp_idx=timepoints.index(timepoint)*2
            idx=offset+tp_idx
            print(idx)
            df.loc[idx]=[int(patient_no)]+["PET"]+[int(timepoint)]+[PET_vol]
            img_type="ME"
            idx+=1
            MRI_img=getMRIImg(patient_no,timepoint,MRI_seg_path)
            if os.path.isfile(MRI_img):
                MRI_vol=getMRIVolume(patient_no,timepoint,MRI_seg_path)
            else:
                MRI_vol=np.NaN
            df.loc[idx]=[int(patient_no)]+[img_type+" MRI"]+[int(timepoint)]+[MRI_vol]
            print(idx)
    return(df)

df=getVolDF(patient_list,timepoints,PET_seg_path,MRI_seg_path)
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volumes_Jan_19_22.csv",index=False)

#df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volumes_Jan_19_22.csv")
PET_df=df[df["IMAGE_TYPE"]=="PET"]
PET_df=PET_df[PET_df["PATIENT_ID"]<19]
PET_df_1=PET_df[PET_df["TIMEPOINT"]==1]
PET_df_2=PET_df[PET_df["TIMEPOINT"]==2]
PET_df_3=PET_df[PET_df["TIMEPOINT"]==3]
PET_tum_vol_1=PET_df_1["TUMOUR VOLUME_CM3"]
PET_tum_vol_1=pd.to_numeric(PET_tum_vol_1)
PET_tum_vol_2=PET_df_2["TUMOUR VOLUME_CM3"]
PET_tum_vol_2=pd.to_numeric(PET_tum_vol_2)
PET_tum_vol_3=PET_df_3["TUMOUR VOLUME_CM3"]
PET_tum_vol_3=pd.to_numeric(PET_tum_vol_3)
PET_tum_vol_1_list=PET_tum_vol_1.values.tolist()
PET_tum_vol_2_list=PET_tum_vol_2.values.tolist()
PET_tum_vol_3_list=PET_tum_vol_3.values.tolist()
#print("PET tumour volume TP 1", PET_tum_vol_1_list)
#print("PET tumour volume TP 2", PET_tum_vol_2_list)
#print("PET tumour volume TP 3", PET_tum_vol_3_list)

relative_PET_volume_change = []

zip_object = zip(PET_tum_vol_1_list, PET_tum_vol_3_list)
for PET_tum_vol_1_list_i, PET_tum_vol_3_list_i in zip_object:
    relative_PET_volume_change.append((PET_tum_vol_3_list_i-PET_tum_vol_1_list_i)/PET_tum_vol_1_list_i)

print("relative PET volume change", relative_PET_volume_change)

MRI_df=df[df["IMAGE_TYPE"]=="ME MRI"]
MRI_df_1=MRI_df[MRI_df["TIMEPOINT"]==1]
MRI_df_2=MRI_df[MRI_df["TIMEPOINT"]==2]
MRI_df_3=MRI_df[MRI_df["TIMEPOINT"]==3]
MRI_tum_vol_1=MRI_df_1["TUMOUR VOLUME_CM3"]
MRI_tum_vol_1=pd.to_numeric(MRI_tum_vol_1)
MRI_tum_vol_2=MRI_df_2["TUMOUR VOLUME_CM3"]
MRI_tum_vol_2=pd.to_numeric(MRI_tum_vol_2)
MRI_tum_vol_3=MRI_df_3["TUMOUR VOLUME_CM3"]
MRI_tum_vol_3=pd.to_numeric(MRI_tum_vol_3)
MRI_tum_vol_1_list=MRI_tum_vol_1.values.tolist()
MRI_tum_vol_2_list=MRI_tum_vol_2.values.tolist()
MRI_tum_vol_3_list=MRI_tum_vol_3.values.tolist()
#print("MRI tumour volume TP 1", MRI_tum_vol_1_list)
#print("MRI tumour volume TP 2", MRI_tum_vol_2_list)
#print("MRI tumour volume TP 3", MRI_tum_vol_3_list)

relative_MRI_volume_change = []

zip_object = zip(MRI_tum_vol_1_list, MRI_tum_vol_3_list)
for MRI_tum_vol_1_list_i, MRI_tum_vol_3_list_i in zip_object:
    relative_MRI_volume_change.append((MRI_tum_vol_3_list_i-MRI_tum_vol_1_list_i)/MRI_tum_vol_1_list_i)

print("relative MRI volume change", relative_MRI_volume_change)

"""
Determining 'responders' and 'non-responders'
"""

#PET_volumes=np.array([0.041935483870967745, -0.4101040118870728, -0.7903525046382189, -0.23210633946830267, -0.5405405405405407, -0.7863720073664825, -0.7274401473296501, -0.659959758551308, -0.558282208588957, -0.5947409126063419, -0.8634435962680237, -0.6400911161731208, -0.6155844155844156])
#DCE_volumes=np.array([-0.9679526136617301, -0.373272496056131, -0.8311915536234014, -0.9326586478639568, -0.9892031708073612, -0.9863581388303398, -0.9628653338910416, -0.8224615377332933, -0.8101386225149402, -0.5749023810786822, -0.8198598760858453, -0.9798479711254512, -0.9834757729418346, 3.5169894164274207, -0.973550563804823, -0.9633848391633426])

PET_volumes_tp1=PET_df[PET_df['TIMEPOINT']==1]['TUMOUR VOLUME_CM3']
PET_volumes_tp3=PET_df[PET_df['TIMEPOINT']==3]['TUMOUR VOLUME_CM3']
PET_tp1_data=PET_volumes_tp1.values
PET_tp3_data=PET_volumes_tp3.values

PET_rel_volume_change=(PET_tp1_data-PET_tp3_data)/PET_tp1_data*100
median_PET=np.median(PET_rel_volume_change)
PET_volumes=[0 if i < median_PET else 1 for i in PET_rel_volume_change]

MRI_volumes_tp1=MRI_df[MRI_df['TIMEPOINT']==1]['TUMOUR VOLUME_CM3']
MRI_volumes_tp3=MRI_df[MRI_df['TIMEPOINT']==3]['TUMOUR VOLUME_CM3']
MRI_tp1_data=MRI_volumes_tp1.values
MRI_tp3_data=MRI_volumes_tp3.values

MRI_rel_volume_change=(MRI_tp1_data-MRI_tp3_data)/MRI_tp1_data*100
median_MRI=np.median(MRI_rel_volume_change)
MRI_volumes=[0 if i < median_MRI else 1 for i in MRI_rel_volume_change]

PET_median=np.median(PET_volumes)
MRI_median=np.median(MRI_volumes)

PET_processed=PET_volumes-PET_median
MRI_processed=MRI_volumes-MRI_median

PET_processed = [0 if i < 0 else 1 for i in PET_processed]
MRI_processed = [0 if i < 0 else 1 for i in MRI_processed]

print("PET processed", PET_processed)
print("MRI processed", MRI_processed)