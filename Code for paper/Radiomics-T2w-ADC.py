#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
from platipy.imaging.registration.utils import smooth_and_resample 
import gc
from platipy.imaging.utils.crop import crop_to_label_extent 

"""
Task: Run radiomics analysis on Before, During, and After PST T2w and ADC images
"""

path="/home/alicja/PET_LAB_PROCESSED/"

def getImgFilepaths(path,patient_no):
    Bef_T2w=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T2W.nii.gz")
    Bef_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T2W_SPAIR.nii.gz")
    Bef_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_DWI_ADC.nii.gz")
    Dur_T2w=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T2W.nii.gz")
    Dur_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T2W_SPAIR.nii.gz")
    Dur_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_DWI_ADC.nii.gz")
    Post_T2w=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T2W.nii.gz")
    Post_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T2W_SPAIR.nii.gz")
    Post_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_DWI_ADC.nii.gz")

    return(Bef_T2w, Bef_T2w_SPAIR, Bef_ADC, Dur_T2w, Dur_T2w_SPAIR, Dur_ADC, Post_T2w, Post_T2w_SPAIR, Post_ADC)

mask_path="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_MRI_tumour_masks/"

def getMaskFilepaths(mask_path,patient_no):
    Bef_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_1_DCE_ACQ_1.nii.gz")
    Dur_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_2_DCE_ACQ_1.nii.gz")
    Post_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_3_DCE_ACQ_1.nii.gz")
    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def radiomics_analysis(image_filepath, mask_filepath,img_label,z_norm="True"):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath)
    
    #Z-score normalisation for MRI
    if z_norm=="True":
        img_arr=sitk.GetArrayFromImage(img)
        img_mean=np.mean(img_arr)
        img_std=np.std(img_arr)
        img=sitk.Cast(img,sitk.sitkInt16)
        img=(img-img_mean)/img_std
    elif z_norm=="False":
        img_arr=sitk.GetArrayFromImage(img)
        img=sitk.Cast(img,sitk.sitkInt16)

    #Grey-level discretisation for MRI    
    # print("Max img value", img_arr.max())
    #img_arr = sitk.GetArrayFromImage(img)
    img_arr = sitk.GetArrayFromImage(sitk.Mask(img,mask))
    # print("Max img value in mask", img_arr.max())
    
    bin_number=512 #256
    min_arr=np.min(img_arr)
    max_arr=np.max(img_arr)
    img_arr[img_arr!=np.max(img_arr)]=np.floor(bin_number*(img_arr[img_arr!=np.max(img_arr)]-min_arr)/(max_arr-min_arr))+1
    img_arr[img_arr==np.max(img_arr)]=bin_number
    
    new_img_arr = img_arr
    new_img=sitk.GetImageFromArray(new_img_arr)
    new_img.CopyInformation(img)
    img=new_img

    ### Isotropic voxel resampling - want MR images to be 0.5 × 0.5 × 0.5 mm3
    img=smooth_and_resample(img,isotropic_voxel_size_mm=0.5)
    mask=sitk.Resample(mask,img)

    img_crop = crop_to_label_extent(img, mask, expansion_mm=10) 
    mask_crop= crop_to_label_extent(mask, mask, expansion_mm=10) 

    #radiomics feature extraction
    extractor = firstorder.RadiomicsFirstOrder(img_crop, mask_crop)
    dict1 = extractor.execute()
    #print("first order features extracted")
    extractor_2 = shape.RadiomicsShape(img_crop, mask_crop)
    dict2 = extractor_2.execute()
    extractor_3 = glcm.RadiomicsGLCM(img_crop, mask_crop)
    dict3 = extractor_3.execute()
    extractor_4 = glszm.RadiomicsGLSZM(img_crop, mask_crop)
    dict4 = extractor_4.execute()
    #print("glszm features extracted")
    extractor_5 = glrlm.RadiomicsGLRLM(img_crop, mask_crop)
    dict5 = extractor_5.execute()
    extractor_6 = ngtdm.RadiomicsNGTDM(img_crop, mask_crop)
    dict6 = extractor_6.execute()
    extractor_7 = gldm.RadiomicsGLDM(img_crop, mask_crop)
    dict7 = extractor_7.execute()
    
    dict1.update(dict2)
    dict1.update(dict3)
    dict1.update(dict4)
    dict1.update(dict5)
    dict1.update(dict6)
    dict1.update(dict7)
    dict1.update({'image label': img_label})

    img=None
    mask=None
    new_img=None
    gc.collect() 

    return(dict1)

#patient_list=("04",)
patient_list=("04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23")

def getImageLabel(img_fp):
    if "SPAIR" in img_fp:
        timepoint=img_fp[-22]
        if timepoint=="1":
            name1="Before PST"
        elif timepoint=="2":
            name1="During PST"
        elif timepoint=="3":
            name1="Post PST"
        name2="T2w SPAIR"
    elif "T2w.nii.gz" in img_fp:
        timepoint=img_fp[-16]
        if timepoint=="1":
            name1="Before PST"
        elif timepoint=="2":
            name1="During PST"
        elif timepoint=="3":
            name1="Post PST"
        name2="T2w"
    elif "ADC" in img_fp:
        timepoint=img_fp[-20]
        if timepoint=="1":
            name1="Before PST"
        elif timepoint=="2":
            name1="During PST"
        elif timepoint=="3":
            name1="Post PST"
        name2="ADC"
    label=str(name1+" "+name2)
    return(label)

df=pd.DataFrame()
for patient_no in patient_list:
    Bef_T2w, Bef_T2w_SPAIR, Bef_ADC, Dur_T2w, Dur_T2w_SPAIR, Dur_ADC, Post_T2w, Post_T2w_SPAIR, Post_ADC=getImgFilepaths(path,patient_no)
    bef_image_fp_list=[Bef_T2w, Bef_T2w_SPAIR, Bef_ADC]
    dur_image_fp_list=[Dur_T2w, Dur_T2w_SPAIR, Dur_ADC]
    post_image_fp_list=[Post_T2w, Post_T2w_SPAIR, Post_ADC]
    Bef_Tumour, Dur_Tumour, Post_Tumour=getMaskFilepaths(mask_path,patient_no)
    for image_filepath in bef_image_fp_list:
        img_label=getImageLabel(image_filepath)
        dict1=radiomics_analysis(image_filepath, Bef_Tumour,img_label)
        #print(dict1)
        dict1.update({'Patient': patient_no})
        df=df.append(dict1,ignore_index=True)
    for image_filepath in dur_image_fp_list:
        img_label=getImageLabel(image_filepath)
        dict2=radiomics_analysis(image_filepath, Dur_Tumour,img_label)
        dict2.update({'Patient': patient_no})
        df=df.append(dict2,ignore_index=True)
    for image_filepath in post_image_fp_list:
        img_label=getImageLabel(image_filepath)
        dict3=radiomics_analysis(image_filepath, Post_Tumour,img_label)
        dict3.update({'Patient': patient_no})
        df=df.append(dict3,ignore_index=True)
    print(f"Patient {patient_no} radiomics extraction complete")

print(df)
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2w-ADC-T2wSPAIR_radiomics_features_Feb_02_22.csv")