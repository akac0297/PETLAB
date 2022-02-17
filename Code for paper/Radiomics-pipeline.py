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

matplotlib.use("Qt4Agg")

"""
Run radiomics analysis on Before, During, and After PST DCE-MRI (Pre-contrast, Post1, Post2, Post3, Post4, Post5, ME), T2w-MRI, ADC (DWI-MRI), and PET images
"""

path="/home/alicja/PET_LAB_PROCESSED/"

def getDCEImgFilepaths(path,patient_no):
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

def getT2wADCImgFilepaths(path,patient_no):
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

mask_path_MRI="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_MRI_tumour_masks/"

def getMRIMaskFilepaths(mask_path_MRI,patient_no):
    Bef_Tumour=str(f"{mask_path_MRI}tumour_mask_WES_0{patient_no}_TIMEPOINT_1_DCE_ACQ_1.nii.gz")
    Dur_Tumour=str(f"{mask_path_MRI}tumour_mask_WES_0{patient_no}_TIMEPOINT_2_DCE_ACQ_1.nii.gz")
    Post_Tumour=str(f"{mask_path_MRI}tumour_mask_WES_0{patient_no}_TIMEPOINT_3_DCE_ACQ_1.nii.gz")
    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def getPETImgFilepaths(path,patient_no):
    prePST_PET=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_1_PET.nii.gz")
    durPST_PET=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_2_PET.nii.gz")
    postPST_PET=str(f"{path}WES_0{patient_no}/IMAGES/WES_0{patient_no}_TIMEPOINT_3_PET.nii.gz")
    return(prePST_PET,durPST_PET,postPST_PET)

mask_path_PET="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/New PET tumours/"

def getPETMaskFilepaths(mask_path_PET,patient_no):
    Bef_Tumour=str(f"{mask_path_PET}WES_0{patient_no}_TIMEPOINT_1_PET_TUMOUR.nii.gz")
    Dur_Tumour=str(f"{mask_path_PET}WES_0{patient_no}_TIMEPOINT_2_PET_TUMOUR.nii.gz")
    Post_Tumour=str(f"{mask_path_PET}WES_0{patient_no}_TIMEPOINT_3_PET_TUMOUR.nii.gz")
    return(Bef_Tumour, Dur_Tumour, Post_Tumour)

def discretise(img,bin_size,max_suv):
    bins = np.arange(0, max_suv, bin_size)

    arr = sitk.GetArrayFromImage(img)
    arr_dig = np.digitize(arr, bins)
    arr_digitized = (arr_dig-1)*bin_size

    img_digitized = sitk.GetImageFromArray(arr_digitized)
    img_digitized.CopyInformation(img) 
    return img_digitized

def radiomics_analysis(image_filepath, mask_filepath,time_label,modality_label,image_type):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath, sitk.sitkUInt8)
    mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkNearestNeighbor)
    
    #Z-score normalisation for MRI (not necessary for PET)
    if image_type == "MRI":
        img_arr=sitk.GetArrayFromImage(img)
        img_mean=np.mean(img_arr)
        img_std=np.std(img_arr)
        img=sitk.Cast(img,sitk.sitkInt16)
        img=(img-img_mean)/img_std
    elif image_type == "PET":
        img=sitk.Cast(img,sitk.sitkInt16)

    #Grey-level discretisation for MRI
    if image_type == "MRI":
        img_arr = sitk.GetArrayFromImage(sitk.Mask(img,mask))
        
        bin_number=512
        min_arr=np.min(img_arr)
        max_arr=np.max(img_arr)
        img_arr[img_arr!=np.max(img_arr)]=np.floor(bin_number*(img_arr[img_arr!=np.max(img_arr)]-min_arr)/(max_arr-min_arr))+1
        img_arr[img_arr==np.max(img_arr)]=bin_number
        
        new_img_arr = img_arr
        new_img=sitk.GetImageFromArray(new_img_arr)
        new_img.CopyInformation(img)
        img=new_img

    #Grey-level discretisation for PET - instead of bin number, use bin size (=0.3)
    if image_type == "PET":
        params={}
        params["binWidth"]=0.3 

    #Isotropic voxel resampling
    if image_type == "MRI":
        # want MR images to be 0.5 × 0.5 × 0.5 mm3
        img=smooth_and_resample(img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkNearestNeighbor)
        mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkNearestNeighbor)
    elif image_type == "PET":
        # want PET images to be 2 × 2 × 2 mm3
        img=smooth_and_resample(img,isotropic_voxel_size_mm=2,interpolator=sitk.sitkNearestNeighbor)
        mask=sitk.Resample(mask,img,sitk.Transform(),sitk.sitkNearestNeighbor)

    #Crop images to speed up processing
    img_crop = crop_to_label_extent(img, mask, expansion_mm=10)
    mask_crop= crop_to_label_extent(mask, mask, expansion_mm=10) 

    #radiomics feature extraction
    if image_type == "MRI":
        extractor = firstorder.RadiomicsFirstOrder(img_crop, mask_crop)
        extractor_2 = shape.RadiomicsShape(img_crop, mask_crop)
        extractor_3 = glcm.RadiomicsGLCM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)
        extractor_4 = glszm.RadiomicsGLSZM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)
        extractor_5 = glrlm.RadiomicsGLRLM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)
        extractor_6 = ngtdm.RadiomicsNGTDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)
        extractor_7 = gldm.RadiomicsGLDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop)
    elif image_type == "PET":
        extractor = firstorder.RadiomicsFirstOrder(img_crop, mask_crop,**params)
        extractor_2 = shape.RadiomicsShape(img_crop, mask_crop,**params)
        extractor_3 = glcm.RadiomicsGLCM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop,**params)
        extractor_4 = glszm.RadiomicsGLSZM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop,**params)
        extractor_5 = glrlm.RadiomicsGLRLM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop,**params)
        extractor_6 = ngtdm.RadiomicsNGTDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop,**params)
        extractor_7 = gldm.RadiomicsGLDM(sitk.Cast(img_crop*10,sitk.sitkInt8), mask_crop,**params)
    dict1 = extractor.execute()
    #print("first order features extracted")
    dict2 = extractor_2.execute()
    dict3 = extractor_3.execute()
    dict4 = extractor_4.execute()
    #print("glszm features extracted")
    dict5 = extractor_5.execute()
    dict6 = extractor_6.execute()
    dict7 = extractor_7.execute()
    
    dict1.update(dict2)
    dict1.update(dict3)
    dict1.update(dict4)
    dict1.update(dict5)
    dict1.update(dict6)
    dict1.update(dict7)
    dict1.update({'time label': time_label})
    dict1.update({'modality': modality_label})

    img=None
    mask=None
    new_img=None
    gc.collect() 

    return(dict1)

def getImageLabel(img_fp):
    if "ACQ" in img_fp:
        timepoint=img_fp[-26]
        DCE_img=img_fp[-8]
        if DCE_img=="0":
            name="Pre"
        else:
            name="Post" + DCE_img
    elif "MPE.nii.gz" in img_fp:
        timepoint=img_fp[-24]
        name="ME"
    elif "SPAIR" in img_fp:
        timepoint=img_fp[-22]
        name2="T2w SPAIR"
    elif "T2W.nii.gz" in img_fp:
        timepoint=img_fp[-16]
        name2="T2w"
    elif "ADC" in img_fp:
        timepoint=img_fp[-20]
        name2="ADC"
    elif "PET" in img_fp:
        timepoint=img_fp[-12]
        name2="PET"
    if timepoint=="1":
        name1="Before PST"
    elif timepoint=="2":
        name1="During PST"
    elif timepoint=="3":
        name1="Post PST"
    else:
        print("Timepoint: ",timepoint,"Image filepath:",img_fp)
    if "ACQ" in img_fp:
        name1 = name1+" "+name
        name2 = "DCE"
    elif "MPE.nii.gz" in img_fp:
        name1 = name1+" "+name
        name2="DCE"
    return(name1, name2)
    #label=str(name1+" "+name2)
    #return(label)

#MRI_patient_list=("04",)
#MRI_patient_list=("04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23")
#patient_list=("04","05","06")
patient_list=("04","05","06","07","08","09","10","12","13","14","15","16","18")

df=pd.DataFrame()
for patient_no in patient_list:
    Bef_Pre,Bef_Post1,Bef_Post2,Bef_Post3,Bef_Post4,Bef_Post5,Bef_ME,Dur_Pre,Dur_Post1,Dur_Post2,Dur_Post3,Dur_Post4,Dur_Post5,Dur_ME,Post_Pre,Post_Post1,Post_Post2,Post_Post3,Post_Post4,Post_Post5,Post_ME=getDCEImgFilepaths(path,patient_no)
    bef_image_fp_list_DCE=[Bef_Pre,Bef_Post1,Bef_Post2,Bef_Post3,Bef_Post4,Bef_Post5,Bef_ME]
    dur_image_fp_list_DCE=[Dur_Pre,Dur_Post1,Dur_Post2,Dur_Post3,Dur_Post4,Dur_Post5,Dur_ME]
    post_image_fp_list_DCE=[Post_Pre,Post_Post1,Post_Post2,Post_Post3,Post_Post4,Post_Post5,Post_ME]

    Bef_T2w, Bef_T2w_SPAIR, Bef_ADC, Dur_T2w, Dur_T2w_SPAIR, Dur_ADC, Post_T2w, Post_T2w_SPAIR, Post_ADC=getT2wADCImgFilepaths(path,patient_no)
    bef_image_fp_list_T2wADC=[Bef_T2w, Bef_T2w_SPAIR, Bef_ADC]
    dur_image_fp_list_T2wADC=[Dur_T2w, Dur_T2w_SPAIR, Dur_ADC]
    post_image_fp_list_T2wADC=[Post_T2w, Post_T2w_SPAIR, Post_ADC]
    Bef_Tumour_MRI, Dur_Tumour_MRI, Post_Tumour_MRI=getMRIMaskFilepaths(mask_path_MRI,patient_no)

    prePST_PET, durPST_PET, postPST_PET=getPETImgFilepaths(path,patient_no)
    Bef_Tumour_PET, Dur_Tumour_PET, Post_Tumour_PET=getPETMaskFilepaths(mask_path_PET,patient_no)

    image_type="MRI"
    for image_filepath in bef_image_fp_list_DCE:
        time_label,modality_label=getImageLabel(image_filepath)
        dict1=radiomics_analysis(image_filepath, Bef_Tumour_MRI,time_label,modality_label,image_type)
        dict1.update({'Patient': patient_no})
        df=df.append(dict1,ignore_index=True)
    for image_filepath in dur_image_fp_list_DCE:
        time_label,modality_label=getImageLabel(image_filepath)
        dict2=radiomics_analysis(image_filepath, Dur_Tumour_MRI,time_label,modality_label,image_type)
        dict2.update({'Patient': patient_no})
        df=df.append(dict2,ignore_index=True)
    for image_filepath in post_image_fp_list_DCE:
        time_label,modality_label=getImageLabel(image_filepath)
        dict3=radiomics_analysis(image_filepath, Post_Tumour_MRI,time_label,modality_label,image_type)
        dict3.update({'Patient': patient_no})
        df=df.append(dict3,ignore_index=True)

    for image_filepath in bef_image_fp_list_T2wADC:
        time_label,modality_label=getImageLabel(image_filepath)
        dict4=radiomics_analysis(image_filepath, Bef_Tumour_MRI,time_label,modality_label,image_type)
        dict4.update({'Patient': patient_no})
        df=df.append(dict4,ignore_index=True)
    for image_filepath in dur_image_fp_list_T2wADC:
        time_label,modality_label=getImageLabel(image_filepath)
        dict5=radiomics_analysis(image_filepath, Dur_Tumour_MRI,time_label,modality_label,image_type)
        dict5.update({'Patient': patient_no})
        df=df.append(dict5,ignore_index=True)
    for image_filepath in post_image_fp_list_T2wADC:
        time_label,modality_label=getImageLabel(image_filepath)
        dict6=radiomics_analysis(image_filepath, Post_Tumour_MRI,time_label,modality_label,image_type)
        dict6.update({'Patient': patient_no})
        df=df.append(dict6,ignore_index=True)

    image_type = "PET"
    time_label,modality_label=getImageLabel(prePST_PET)
    dict7=radiomics_analysis(prePST_PET, Bef_Tumour_PET,time_label,modality_label,image_type)
    dict7.update({'Patient': patient_no})
    df=df.append(dict7,ignore_index=True)

    time_label,modality_label=getImageLabel(durPST_PET)
    dict8=radiomics_analysis(durPST_PET, Dur_Tumour_PET,time_label,modality_label,image_type)
    dict8.update({'Patient': patient_no})
    df=df.append(dict8,ignore_index=True)

    time_label,modality_label=getImageLabel(postPST_PET)
    dict9=radiomics_analysis(postPST_PET, Post_Tumour_PET,time_label,modality_label,image_type)
    dict9.update({'Patient': patient_no})
    df=df.append(dict9,ignore_index=True)

    print(f"Patient {patient_no} radiomics extraction complete")

print(df)
df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_Feb_17_22.csv")