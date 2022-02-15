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

def getPETseg(image_pt_0_1,image_pt_0_2,image_pt_0_3,patient_no,timepoints,masked_breast_path):
    for timepoint in timepoints:
        if timepoint == "1":
            if patient_no=="14":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_new.nii.gz")
            elif patient_no=="15" and timepoint=="1":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
            else:
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+".nii.gz")
            masked_pet_breast=sitk.Resample(masked_pet_breast,image_pt_0_1)
            mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
            mask_arr=mask_arr.flatten()
            suv_max=np.max(mask_arr)
            t=0.4*suv_max
            tum1 = sitk.Mask(image_pt_0_1, masked_pet_breast>t)
            tum1 = sitk.Cast(tum1, sitk.sitkInt64)
            tum1_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum1))
            tum1 = (tum1_cc==1)
        elif timepoint == "2":
            if patient_no=="14":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_new.nii.gz")
            elif patient_no=="19" and timepoint=="2":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
            else:
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+".nii.gz")
            masked_pet_breast=sitk.Resample(masked_pet_breast,image_pt_0_2)
            tum2 = sitk.Mask(image_pt_0_2, masked_pet_breast>t)
            tum2 = sitk.Cast(tum2, sitk.sitkInt64)
            tum2_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum2))
            tum2 = (tum2_cc==1)
        elif timepoint == "3":
            if patient_no=="12" and timepoint=="3":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_old.nii.gz")
            elif patient_no=="14":
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+"_new.nii.gz")
            else:
                masked_pet_breast=sitk.ReadImage(masked_breast_path+"masked_pet_breast_WES_0"+patient_no+"_"+str(int(timepoint)-1)+".nii.gz")
            masked_pet_breast=sitk.Resample(masked_pet_breast,image_pt_0_3)
            tum3 = sitk.Mask(image_pt_0_3, masked_pet_breast>t)
            tum3 = sitk.Cast(tum3, sitk.sitkInt64)
            tum3_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum3))
            tum3 = (tum3_cc==1)
    return(tum1,tum2,tum3)

masked_breast_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/"
path="/home/alicja/PET_LAB_PROCESSED/"
patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19"]
timepoints=["1","2","3"]

for patient_no in patient_list:
    image_pt_0_1=getPETimage(patient_no,"1",path)
    image_pt_0_2=getPETimage(patient_no,"2",path)
    image_pt_0_3=getPETimage(patient_no,"3",path)
    tum1,tum2,tum3=getPETseg(image_pt_0_1,image_pt_0_2,image_pt_0_3,patient_no,timepoints,masked_breast_path)
    output_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET_40pc_SUVmax_tumours/"
    sitk.WriteImage(tum1, output_path+"WES_0"+patient_no+"_TIMEPOINT_1_PET_TUMOUR.nii.gz")
    print(f"Patient {patient_no} timepoint 1 PET tumour complete")
    sitk.WriteImage(tum2, output_path+"WES_0"+patient_no+"_TIMEPOINT_2_PET_TUMOUR.nii.gz")
    print(f"Patient {patient_no} timepoint 2 PET tumour complete")
    sitk.WriteImage(tum3, output_path+"WES_0"+patient_no+"_TIMEPOINT_3_PET_TUMOUR.nii.gz")
    print(f"Patient {patient_no} timepoint 3 PET tumour complete")
