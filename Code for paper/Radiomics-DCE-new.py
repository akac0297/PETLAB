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
Task: Run radiomics analysis on Before, During, and After PST DCE images (Pre-contrast, Post1, Post2, Post3, Post4, Post5)
"""

path="/home/alicja/PET_LAB_PROCESSED/"

def getImgFilepaths(path,patient_no):
    Bef_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_0.nii.gz")
    Bef_Post1=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_1.nii.gz")
    Bef_Post2=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_2.nii.gz")
    Bef_Post3=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_3.nii.gz")
    Bef_Post4=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_4.nii.gz")
    Bef_Post5=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_5.nii.gz")
    Bef_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_MPE.nii.gz")

    Dur_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_0.nii.gz")
    Dur_Post1=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_1.nii.gz")
    Dur_Post2=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_2.nii.gz")
    Dur_Post3=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_3.nii.gz")
    Dur_Post4=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_4.nii.gz")
    Dur_Post5=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_5.nii.gz")
    Dur_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_MPE.nii.gz")

    Post_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_0.nii.gz")
    Post_Post1=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_1.nii.gz")
    Post_Post2=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_2.nii.gz")
    Post_Post3=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_3.nii.gz")
    Post_Post4=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_4.nii.gz")
    Post_Post5=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_5.nii.gz")
    Post_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_MPE.nii.gz")

    return(Bef_Pre,Bef_Post1,Bef_Post2,Bef_Post3,Bef_Post4,Bef_Post5,Bef_ME,Dur_Pre,Dur_Post1,Dur_Post2,Dur_Post3,Dur_Post4,Dur_Post5,Dur_ME,
    Post_Pre,Post_Post1,Post_Post2,Post_Post3,Post_Post4,Post_Post5,Post_ME)

mask_path="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_MRI_tumour_masks/"

def getMaskFilepaths(mask_path,patient_no):
    Bef_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_1_DCE_ACQ_1.nii.gz")
    Dur_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_2_DCE_ACQ_1.nii.gz")
    Post_Tumour=str(f"{mask_path}tumour_mask_WES_0{patient_no}_TIMEPOINT_3_DCE_ACQ_1.nii.gz")
    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def radiomics_analysis(image_filepath, mask_filepath,img_label,z_norm="True"):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath)
    mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkNearestNeighbor)
    
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

patient_list=("05",)
#patient_list=("04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23")

def getImageLabel(img_fp):
    if "MPE" not in img_fp:
        timepoint=img_fp[-26]
        if timepoint=="1":
            name1="Before PST"
        elif timepoint=="2":
            name1="During PST"
        elif timepoint=="3":
            name1="Post PST"
        DCE_img=img_fp[-8]
        if DCE_img=="0":
            name2="Pre"
        elif DCE_img=="1":
            name2="Post1"
        elif DCE_img=="2":
            name2="Post2"
        elif DCE_img=="3":
            name2="Post3"
        elif DCE_img=="4":
            name2="Post4"
        elif DCE_img=="5":
            name2="Post5"
    else:
        timepoint=img_fp[-24]
        if timepoint=="1":
            name1="Before PST"
        elif timepoint=="2":
            name1="During PST"
        elif timepoint=="3":
            name1="Post PST"
        name2="ME"
    label=str(name1+" "+name2)
    return(label)

df=pd.DataFrame()
for patient_no in patient_list:
    Bef_Pre,Bef_Post1,Bef_Post2,Bef_Post3,Bef_Post4,Bef_Post5,Bef_ME,Dur_Pre,Dur_Post1,Dur_Post2,Dur_Post3,Dur_Post4,Dur_Post5,Dur_ME,Post_Pre,Post_Post1,Post_Post2,Post_Post3,Post_Post4,Post_Post5,Post_ME=getImgFilepaths(path,patient_no)
    bef_image_fp_list=[Bef_Pre,Bef_Post1,Bef_Post2,Bef_Post3,Bef_Post4,Bef_Post5,Bef_ME]
    dur_image_fp_list=[Dur_Pre,Dur_Post1,Dur_Post2,Dur_Post3,Dur_Post4,Dur_Post5,Dur_ME]
    post_image_fp_list=[Post_Pre,Post_Post1,Post_Post2,Post_Post3,Post_Post4,Post_Post5,Post_ME]
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
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features_Feb_04_22_WES_005.csv")