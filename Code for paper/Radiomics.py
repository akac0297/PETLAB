#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
from platipy.imaging.registration.utils import smooth_and_resample 
import gc
from platipy.imaging.utils.crop import crop_to_label_extent 

matplotlib.use("Qt5Agg")

"""
Run radiomics analysis on Before, During, and After PST DCE-MRI (ME), T1w-MRI (DCE Pre-contrast) T2w-MRI, ADC (DWI-MRI) images - contralateral breast
"""

path="/home/alicja/PET_LAB_PROCESSED/"

def getT1wImgFilepaths(path,patient_no):
    Bef_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_ACQ_0.nii.gz")
    Dur_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_ACQ_0.nii.gz")
    Post_Pre=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_ACQ_0.nii.gz")

    return(Bef_Pre,Dur_Pre,Post_Pre)

def getMEImgFilepaths(path,patient_no):
    Bef_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T1W_DCE_MPE.nii.gz")
    Dur_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T1W_DCE_MPE.nii.gz")
    Post_ME=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T1W_DCE_MPE.nii.gz")

    return(Bef_ME,Dur_ME,Post_ME)

def getT2wImgFilepaths(path,patient_no):
    Bef_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_T2W_SPAIR.nii.gz")
    Dur_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_T2W_SPAIR.nii.gz")
    Post_T2w_SPAIR=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_T2W_SPAIR.nii.gz")

    return(Bef_T2w_SPAIR, Dur_T2w_SPAIR, Post_T2w_SPAIR)

def getADCImgFilepaths(path,patient_no):
    Bef_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_MRI_DWI_ADC.nii.gz")
    Dur_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_MRI_DWI_ADC.nii.gz")
    Post_ADC=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_MRI_DWI_ADC.nii.gz")

    return(Bef_ADC, Dur_ADC, Post_ADC)

mask_path_MRI="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/PETLAB Breast Masks/Contralateral-breasts/"
#mask_path_MRI="/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/"

# pre-processing
# (no normalisation & z-score normalisation) for T2w, T1w images
# no z-score normalisation for ADC or DCE

def getMRIMaskFilepaths(mask_path_MRI,patient_no):
    # This is for whole breast radiomics
    Bef_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_1_T2W_contralateral.nii.gz")
    Dur_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_2_T2W_contralateral.nii.gz")
    Post_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_3_T2W_contralateral.nii.gz")

    # # This is for tumour radiomics
    # Bef_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}/STRUCTURES/WES_0{patient_no}_TIMEPOINT_1_GTV.nii.gz")
    # Dur_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}/STRUCTURES/WES_0{patient_no}_TIMEPOINT_2_GTV.nii.gz")
    # Post_Tumour=str(f"{mask_path_MRI}WES_0{patient_no}/STRUCTURES/WES_0{patient_no}_TIMEPOINT_3_GTV.nii.gz")

    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def radiomics_analysis(image_filepath, mask_filepath,time_label,modality_label,normalisation,bincount):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath, sitk.sitkUInt8)

    #Isotropic voxel resampling - want MR images to be 0.5 × 0.5 × 0.5 mm3
    img=smooth_and_resample(img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear) # want to resample the image instead of the mask
    mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkLinear)
    # mask=smooth_and_resample(mask,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)
    # img=sitk.Resample(img,mask,sitk.Transform(),sitk.sitkLinear)

    # print(mask.GetSize(), img.GetSize())
    # print(mask.GetSpacing(),img.GetSpacing())

    # sitk.WriteImage(img,"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/img_with_linear_interpolation.nii.gz")
    # print("Image saved")

    print("Image and mask resampled")
    
    #Z-score normalisation -- only need to do this for T2w and T1w images
    if normalisation==True: #("T2W" or "ACQ_0") in image_filepath
        img_arr=sitk.GetArrayFromImage(img)
        img_mean=np.mean(img_arr)
        img_std=np.std(img_arr)
        img=sitk.Cast(img,sitk.sitkInt16)
        img=(img-img_mean)/img_std
        # sitk.WriteImage(img,"/home/alicja/Documents/Z-score-normalised-img.nii.gz")
        print("T2w or T1w image normalised")

    #Crop images to speed up processing
    img_crop = crop_to_label_extent(img, mask, expansion_mm=10)
    mask_crop= crop_to_label_extent(mask, mask, expansion_mm=10) 
    print("image and mask cropped")
    # sitk.WriteImage(img_crop, "/home/alicja/Documents/cropped-img-WES0{patient_no}.nii.gz")
    # sitk.WriteImage(mask_crop, "/home/alicja/Documents/cropped-mask-WES0{patient_no}.nii.gz")
    # print("image and mask saved")

    #radiomics feature extraction
    extractor = firstorder.RadiomicsFirstOrder(img_crop, mask_crop)#, binCount=bincount) 
    #binCount does grey-level discretisation of the images
    print("extractor 1 generated")
    extractor_2 = shape.RadiomicsShape(img_crop, mask_crop)#, binCount=bincount)
    print("extractor 2 generated")
    extractor_3 = glcm.RadiomicsGLCM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)#, binCount=bincount)
    print("extractor 3 generated")
    extractor_4 = glszm.RadiomicsGLSZM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)#, binCount=bincount)
    print("extractor 4 generated")
    extractor_5 = glrlm.RadiomicsGLRLM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)#, binCount=bincount)
    print("extractor 5 generated")
    extractor_6 = ngtdm.RadiomicsNGTDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)#, binCount=bincount)
    print("extractor 6 generated")
    extractor_7 = gldm.RadiomicsGLDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)#, binCount=bincount)
    print("extractors generated")
    dict1 = extractor.execute()
    print("first order features extracted")
    dict2 = extractor_2.execute()
    print("shape features extracted")
    dict3 = extractor_3.execute()
    print("glcm features extracted")
    dict4 = extractor_4.execute()
    print("glszm features extracted")
    dict5 = extractor_5.execute()
    print("glrlm features extracted")
    dict6 = extractor_6.execute()
    print("ngtdm features extracted")
    dict7 = extractor_7.execute()
    print("gldm features extracted")
    
    df1=pd.Series(dict1)
    df2=pd.Series(dict2)
    df3=pd.Series(dict3)
    df4=pd.Series(dict4)
    df5=pd.Series(dict5)
    df6=pd.Series(dict6)
    df7=pd.Series(dict7)

    df=pd.concat([df1,df2,df3,df4,df5,df6,df7],axis=0)
    df["time label"] = time_label
    df["modality"] = modality_label

    img=None
    mask=None
    extractor_4=None
    dict4=None
    gc.collect() 

    return(df)

def getImageLabel(img_fp):
    if "ACQ" in img_fp:
        timepoint=img_fp[-26]
        DCE_img=img_fp[-8]
        if DCE_img=="0":
            modality_label="T1w"
    elif "MPE.nii.gz" in img_fp:
        timepoint=img_fp[-24]
        modality_label="DCE ME"
    elif "SPAIR" in img_fp:
        timepoint=img_fp[-22]
        modality_label="T2w SPAIR"
    elif "T2W.nii.gz" in img_fp:
        timepoint=img_fp[-16]
        modality_label="T2w"
    elif "ADC" in img_fp:
        timepoint=img_fp[-20]
        modality_label="ADC"
    if timepoint=="1":
        time_label="Before PST"
    elif timepoint=="2":
        time_label="During PST"
    elif timepoint=="3":
        time_label="Post PST"
    else:
        print("Timepoint: ",timepoint,"Image filepath:",img_fp)
    return(time_label, modality_label)

MRI_patient_list=("16",)
#MRI_patient_list=("06","14","15","16","18","19","21","23")
df=pd.DataFrame()

# log_file = f'/home/alicja/PET-LAB Code/PET-LAB/debug-file-Patient-19.txt'
# from radiomics import logging, logger, setVerbosity
# handler = logging.FileHandler(filename=log_file, mode='w')  # overwrites log_files from previous runs. Change mode to 'a' to append.
# formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")  # format string for log messages
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# setVerbosity(logging.DEBUG)

# # Control the amount of logging stored by setting the level of the logger. N.B. if the level is higher than the
# # Verbositiy level, the logger level will also determine the amount of information printed to the output
# logger.setLevel(logging.DEBUG)

for patient_no in MRI_patient_list:
    Bef_Pre,Dur_Pre,Post_Pre=getT1wImgFilepaths(path,patient_no)
    Bef_ME,Dur_ME,Post_ME=getMEImgFilepaths(path,patient_no)
    Bef_T2w_SPAIR, Dur_T2w_SPAIR, Post_T2w_SPAIR=getT2wImgFilepaths(path,patient_no)
    Bef_ADC,Dur_ADC,Post_ADC=getADCImgFilepaths(path,patient_no)
    Bef_Tumour_MRI, Dur_Tumour_MRI, Post_Tumour_MRI=getMRIMaskFilepaths(mask_path_MRI,patient_no)

    T1w_T2w_Bef_list=[Bef_Pre, Bef_T2w_SPAIR]
    T1w_T2w_Dur_list=[Dur_Pre,Dur_T2w_SPAIR]
    T1w_T2w_Post_list=[Post_Pre,Post_T2w_SPAIR]
    ME_ADC_Bef_list=[Bef_ME,Bef_ADC]
    ME_ADC_Dur_list=[Dur_ME,Dur_ADC]
    ME_ADC_Post_list=[Post_ME,Post_ADC]

    binCount=256

    for image_filepath in T1w_T2w_Bef_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Bef_Tumour_MRI,time_label,modality_label,True,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "True"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in T1w_T2w_Dur_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Dur_Tumour_MRI,time_label,modality_label,True,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "True"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in T1w_T2w_Post_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Post_Tumour_MRI,time_label,modality_label,True,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "True"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in T1w_T2w_Bef_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Bef_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in T1w_T2w_Dur_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Dur_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in T1w_T2w_Post_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Post_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in ME_ADC_Bef_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Bef_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in ME_ADC_Dur_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Dur_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()
    for image_filepath in ME_ADC_Post_list:
        time_label,modality_label=getImageLabel(image_filepath)
        df_features=radiomics_analysis(image_filepath, Post_Tumour_MRI,time_label,modality_label,False,binCount)
        df_features['Patient'] = patient_no
        df_features['Z-score normalisation'] = "False"
        df=df.append(df_features,ignore_index=True)
        df_features=None
        gc.collect()

    print(f"Patient {patient_no} radiomics extraction complete")

print(df)
df.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_016_binCount{binCount}_df.csv")