#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np

def getPETimage(patient_no,timepoint,path):
    folder="WES_0"+patient_no+"/IMAGES/"
    ct="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_CT_AC.nii.gz"
    pet="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET.nii.gz"
    image_ct_0=sitk.ReadImage(path+folder+ct)
    image_pt_0_raw=sitk.ReadImage(path+folder+pet)
    image_pt_0=sitk.Resample(image_pt_0_raw, image_ct_0)
    return(image_pt_0)

def getPETseg(image_pt_0,patient_no,timepoint,masked_breast_path):
    if patient_no=="12" and timepoint=="3":
        masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
    elif patient_no=="14":
        masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_new.nii.gz")
    elif patient_no=="15" and timepoint=="1":
        masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
    elif patient_no=="19" and timepoint=="2":
        masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
    else:
        masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+".nii.gz")
    masked_pet_breast=sitk.Resample(masked_pet_breast,image_pt_0)
    print("Size of resampled PET breast:",masked_pet_breast.GetSize())
    print("Size of PET image:", image_pt_0.GetSize())
    mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
    mask_arr=mask_arr.flatten()
    suv_max=np.max(mask_arr)
    t=0.4*suv_max
    tum = sitk.Mask(image_pt_0, masked_pet_breast>t)
    tum = sitk.Cast(tum, sitk.sitkInt64)
    tum_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum))
    tum = (tum_cc==1)
    return(tum)

masked_breast_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/"
path="/home/alicja/PET_LAB_PROCESSED/"
patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19"]
timepoints=["1","2","3"]

for patient_no in patient_list:
    for timepoint in timepoints:
        image_pt_0=getPETimage(patient_no,timepoint,path)
        tum=getPETseg(image_pt_0,patient_no,timepoint,masked_breast_path)
        output_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET_40pc_SUVmax_tumours/"
        sitk.WriteImage(tum, output_path+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET_TUMOUR.nii.gz")
        print(f"Patient {patient_no} timepoint {timepoint} PET tumour complete")