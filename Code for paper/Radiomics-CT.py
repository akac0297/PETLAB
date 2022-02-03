#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm

from platipy.imaging.registration.utils import smooth_and_resample
from platipy.imaging.utils.crop import crop_to_label_extent 

matplotlib.use("Qt4Agg")

"""
This program runs radiomics analysis on Before, During, and After PST CT images (from FDG PET-CT)
"""

def discretise(img,bin_size,max_suv): #check if we need to discretise CT images in the same way as PET
    bins = np.arange(0, max_suv, bin_size)

    arr = sitk.GetArrayFromImage(img)
    arr_dig = np.digitize(arr, bins)
    arr_digitized = (arr_dig-1)*bin_size

    img_digitized = sitk.GetImageFromArray(arr_digitized)
    img_digitized.CopyInformation(img) 
    return img_digitized

path="/home/alicja/PET_LAB_PROCESSED/"

def getImgFilepaths(path,patient_no):
    prePST_CT=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_CT_AC.nii.gz")
    durPST_CT=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_CT_AC.nii.gz")
    postPST_CT=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_CT_AC.nii.gz")
    return(prePST_CT,durPST_CT,postPST_CT)

mask_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/New PET tumours/"

def getMaskFilepaths(mask_path,patient_no):
    Bef_Tumour=str(f"{mask_path}WES_0{patient_no}_TIMEPOINT_1_PET_TUMOUR.nii.gz")
    Dur_Tumour=str(f"{mask_path}WES_0{patient_no}_TIMEPOINT_2_PET_TUMOUR.nii.gz")
    Post_Tumour=str(f"{mask_path}WES_0{patient_no}_TIMEPOINT_3_PET_TUMOUR.nii.gz")
    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def radiomics_analysis(image_filepath, mask_filepath,img_label):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath,sitk.sitkUInt8)
    
    #No need to do Z-score normalisation for CT

    #Isotropic voxel resampling - want images to be 2 × 2 × 2 mm3 ///////// Do we need this for CT as well?
    img=smooth_and_resample(img,isotropic_voxel_size_mm=2,interpolator=sitk.sitkNearestNeighbor)
    mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkNearestNeighbor)

    #Crop images to speed up processing
    new_img=img
    img_crop = crop_to_label_extent(img, mask, expansion_mm=10)
    new_img_crop = crop_to_label_extent(new_img, mask, expansion_mm=10) 
    mask_crop= crop_to_label_extent(mask, mask, expansion_mm=10) 

    #Grey-level discretisation - instead of bin number we use bin size (=0.3) ////////// Do we need this for CT as well?
    params={}
    params["binWidth"]=0.3 

    #radiomics feature extraction
    extractor = firstorder.RadiomicsFirstOrder(img_crop, mask_crop,**params)
    dict1 = extractor.execute()
    #print("first order features extracted")
    extractor_2 = shape.RadiomicsShape(new_img_crop, mask_crop,**params)
    dict2 = extractor_2.execute()
    extractor_3 = glcm.RadiomicsGLCM(sitk.Cast(new_img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict3 = extractor_3.execute()
    extractor_4 = glszm.RadiomicsGLSZM(sitk.Cast(new_img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict4 = extractor_4.execute()
    #print("glszm features extracted")
    extractor_5 = glrlm.RadiomicsGLRLM(sitk.Cast(new_img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict5 = extractor_5.execute()
    extractor_6 = ngtdm.RadiomicsNGTDM(sitk.Cast(new_img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict6 = extractor_6.execute()
    extractor_7 = gldm.RadiomicsGLDM(sitk.Cast(new_img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict7 = extractor_7.execute()
    
    dict1.update(dict2)
    dict1.update(dict3)
    dict1.update(dict4)
    dict1.update(dict5)
    dict1.update(dict6)
    dict1.update(dict7)
    dict1.update({'image label': img_label})

    return(dict1)

patient_list=("04","05","06","07","08","09","10","12","13","14","15","16","18")

def getImageLabel(img_fp):
    timepoint=img_fp[-12]
    if timepoint=="1":
        label="Before-PST CT"
    elif timepoint=="2":
        label="During-PST CT"
    elif timepoint=="3":
        label="Post-PST CT"
    return(label)

df=pd.DataFrame()
for patient_no in patient_list:
    prePST_CT,durPST_CT, postPST_CT=getImgFilepaths(path,patient_no)
    Bef_Tumour, Dur_Tumour, Post_Tumour=getMaskFilepaths(mask_path,patient_no)

    img_label=getImageLabel(prePST_CT)
    dict1=radiomics_analysis(prePST_CT, Bef_Tumour,img_label)
    dict1.update({'Patient': patient_no})
    df=df.append(dict1,ignore_index=True)

    img_label=getImageLabel(durPST_CT)
    dict2=radiomics_analysis(durPST_CT, Dur_Tumour,img_label)
    dict2.update({'Patient': patient_no})
    df=df.append(dict2,ignore_index=True)

    img_label=getImageLabel(postPST_CT)
    dict3=radiomics_analysis(postPST_CT, Post_Tumour,img_label)
    dict3.update({'Patient': patient_no})
    df=df.append(dict3,ignore_index=True)

    print(f"Patient {patient_no} radiomics extraction complete")

print(df)
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/CT_radiomics_features_Feb_02_22.csv")