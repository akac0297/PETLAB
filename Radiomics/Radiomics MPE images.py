#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import pandas as pd
import numpy as np
import radiomics

def radiomics_analysis(image_filepath, mask_filepath,img_label):
    img = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath)
    
    #Z-score normalisation for MRI
    if img_label==("B50T_CAD_ADC_3000" or "B50_800 ADC" or "MPE"):
        z_norm=False
    else:
        z_norm=True
    
    if z_norm==True:
        img_arr=sitk.GetArrayFromImage(img)
        img_mean=np.mean(img_arr)
        img_std=np.std(img_arr)
        img=sitk.Cast(img,sitk.sitkInt16)
        img=(img-img_mean)/img_std
    elif z_norm==False:
        img_arr=sitk.GetArrayFromImage(img)
        img=sitk.Cast(img,sitk.sitkInt16)

    #Grey-level discretisation for MRI    
    img_arr = sitk.GetArrayFromImage(img)  
    mask=sitk.Resample(mask,img)
    
    bin_number=512
    min_arr=np.min(img_arr)
    max_arr=np.max(img_arr)
    img_arr[img_arr!=np.max(img_arr)]=np.floor(bin_number*(img_arr[img_arr!=np.max(img_arr)]-min_arr)/(max_arr-min_arr))+1
    img_arr[img_arr==np.max(img_arr)]=bin_number
    
    new_img_arr = img_arr
    new_img=sitk.GetImageFromArray(new_img_arr)
    new_img.CopyInformation(img)
    img=new_img
    
    extractor = radiomics.firstorder.RadiomicsFirstOrder(img, mask)
    dict1 = extractor.execute()
    extractor_2 = radiomics.shape.RadiomicsShape(img, mask)
    dict2 = extractor_2.execute()
    extractor_3 = radiomics.glcm.RadiomicsGLCM(img, mask)
    dict3 = extractor_3.execute()
    extractor_4 = radiomics.glszm.RadiomicsGLSZM(img, mask)
    dict4 = extractor_4.execute()
    extractor_5 = radiomics.glrlm.RadiomicsGLRLM(img, mask)
    dict5 = extractor_5.execute()
    extractor_6 = radiomics.ngtdm.RadiomicsNGTDM(img, mask)
    dict6 = extractor_6.execute()
    extractor_7 = radiomics.gldm.RadiomicsGLDM(img, mask)
    dict7 = extractor_7.execute()
    
    dict1.update(dict2)
    dict1.update(dict3)
    dict1.update(dict4)
    dict1.update(dict5)
    dict1.update(dict6)
    dict1.update(dict7)
    new_img_label=img_label
    if img_label=="B50T_CAD_ADC_3000":
        new_img_label="B50T_CAD_ADC_3000 no norm"
    if img_label=="B50_800 ADC":
        new_img_label="B50_800 ADC no norm"
    dict1.update({'image label': new_img_label})

    return(dict1)

mask_list=['new_seg_003_2_mri.nii.gz', 'new_seg_004_4_mri.nii.gz', 'new_seg_004_5_mri.nii.gz', 'new_seg_004_6_mri.nii.gz', 'new_seg_005_4_mri.nii.gz', 'new_seg_005_5_mri.nii.gz', 'new_seg_005_6_mri.nii.gz', 'new_seg_006_4_mri.nii.gz', 'new_seg_006_5_mri.nii.gz', 'new_seg_006_6_mri.nii.gz', 'new_seg_007_4_mri.nii.gz', 'new_seg_007_5_mri.nii.gz', 'new_seg_007_6_mri.nii.gz', 'new_seg_008_4_mri.nii.gz', 'new_seg_008_5_mri.nii.gz', 'new_seg_008_6_mri.nii.gz', 'new_seg_009_6_mri.nii.gz', 'new_seg_009_7_mri.nii.gz', 'new_seg_009_8_mri.nii.gz', 'new_seg_010_4_mri.nii.gz', 'new_seg_010_5_mri.nii.gz', 'new_seg_010_6_mri.nii.gz', 'new_seg_012_4_mri.nii.gz', 'new_seg_012_5_mri.nii.gz', 'new_seg_012_6_mri.nii.gz', 'new_seg_013_4_mri.nii.gz', 'new_seg_013_5_mri.nii.gz', 'new_seg_013_6_mri.nii.gz', 'new_seg_014_4_mri.nii.gz', 'new_seg_014_5_mri.nii.gz', 'new_seg_014_6_mri.nii.gz', 'new_seg_015_4_mri.nii.gz', 'new_seg_015_5_mri.nii.gz', 'new_seg_015_6_mri.nii.gz', 'new_seg_016_3_mri.nii.gz', 'new_seg_016_4_mri.nii.gz', 'new_seg_016_5_mri.nii.gz', 'new_seg_017_3_mri.nii.gz', 'new_seg_018_4_mri.nii.gz', 'new_seg_018_5_mri.nii.gz', 'new_seg_018_6_mri.nii.gz', 'new_seg_019_3_mri.nii.gz', 'new_seg_019_4_mri.nii.gz', 'new_seg_019_5_mri.nii.gz', 'new_seg_021_2_mri.nii.gz', 'new_seg_021_3_mri.nii.gz', 'new_seg_021_4_mri.nii.gz', 'new_seg_023_2_mri.nii.gz', 'new_seg_023_3_mri.nii.gz', 'new_seg_023_4_mri.nii.gz', 'new_seg_024_3_mri.nii.gz', 'new_seg_024_4_mri.nii.gz', 'new_seg_024_5_mri.nii.gz']
MPE_list=['max_img_WES_003_2.nii.gz', 'MPE_sub_WES_004_4.nii.gz', 'MPE_sub_WES_004_5.nii.gz', 'MPE_sub_WES_004_6.nii.gz', 'MPE_sub_WES_005_4.nii.gz', 'MPE_sub_WES_005_5.nii.gz', 'MPE_sub_WES_005_6.nii.gz', 'max_img_WES_006_4.nii.gz', 'max_img_WES_006_5.nii.gz', 'max_img_WES_006_6.nii.gz', 'max_img_WES_007_4.nii.gz', 'max_img_WES_007_5.nii.gz', 'max_img_WES_007_6.nii.gz', 'MPE_sub_WES_008_4.nii.gz', 'MPE_sub_WES_008_5.nii.gz', 'MPE_sub_WES_008_6.nii.gz', 'MPE_sub_WES_009_6.nii.gz', 'MPE_sub_WES_009_7.nii.gz', 'MPE_sub_WES_009_8.nii.gz', 'MPE_sub_WES_010_4.nii.gz', 'MPE_sub_WES_010_5.nii.gz', 'MPE_sub_WES_010_6.nii.gz', 'MPE_sub_WES_012_4.nii.gz', 'MPE_sub_WES_012_5.nii.gz', 'MPE_sub_WES_012_6.nii.gz', 'max_img_WES_013_4.nii.gz', 'max_img_WES_013_5.nii.gz', 'max_img_WES_013_6.nii.gz', 'max_img_WES_014_4.nii.gz', 'max_img_WES_014_5.nii.gz', 'max_img_WES_014_6.nii.gz', 'max_img_WES_015_4.nii.gz', 'max_img_WES_015_5.nii.gz', 'max_img_WES_015_6.nii.gz', 'max_img_WES_016_3.nii.gz', 'max_img_WES_016_4.nii.gz', 'max_img_WES_016_5.nii.gz', 'max_img_WES_017_3.nii.gz', 'max_img_WES_018_4.nii.gz', 'max_img_WES_018_5.nii.gz', 'max_img_WES_018_6.nii.gz', 'max_img_WES_019_3.nii.gz', 'max_img_WES_019_4.nii.gz', 'max_img_WES_019_5.nii.gz', 'max_img_WES_021_2.nii.gz', 'max_img_WES_021_3.nii.gz', 'max_img_WES_021_4.nii.gz', 'max_img_WES_023_2.nii.gz', 'max_img_WES_023_3.nii.gz', 'max_img_WES_023_4.nii.gz', 'max_img_WES_024_3.nii.gz', 'max_img_WES_024_4.nii.gz', 'max_img_WES_024_5.nii.gz']
sphere_list=['image_sphere_WES_003_2.nii.gz' 'image_sphere_WES_004_4.nii.gz', 'image_sphere_WES_004_5.nii.gz', 'image_sphere_WES_004_6.nii.gz', 'image_sphere_WES_005_4.nii.gz', 'image_sphere_WES_005_5.nii.gz', 'image_sphere_WES_005_6.nii.gz', 'image_sphere_WES_006_4.nii.gz', 'image_sphere_WES_006_5.nii.gz', 'image_sphere_WES_006_6.nii.gz', 'image_sphere_WES_007_4.nii.gz', 'image_sphere_WES_007_5.nii.gz', 'image_sphere_WES_007_6.nii.gz', 'image_sphere_WES_008_4.nii.gz', 'image_sphere_WES_008_5.nii.gz', 'image_sphere_WES_008_6.nii.gz', 'image_sphere_WES_009_6.nii.gz', 'image_sphere_WES_009_7.nii.gz', 'image_sphere_WES_009_8.nii.gz', 'image_sphere_WES_010_4.nii.gz', 'image_sphere_WES_010_5.nii.gz', 'image_sphere_WES_010_6.nii.gz', 'image_sphere_WES_012_4.nii.gz', 'image_sphere_WES_012_5.nii.gz', 'image_sphere_WES_012_6.nii.gz', 'image_sphere_WES_013_4.nii.gz', 'image_sphere_WES_013_5.nii.gz', 'image_sphere_WES_013_6.nii.gz', 'image_sphere_WES_014_4.nii.gz', 'image_sphere_WES_014_5.nii.gz', 'image_sphere_WES_014_6.nii.gz', 'image_sphere_WES_015_4.nii.gz', 'image_sphere_WES_015_5.nii.gz', 'image_sphere_WES_015_6.nii.gz', 'image_sphere_WES_016_3.nii.gz', 'image_sphere_WES_016_4.nii.gz', 'image_sphere_WES_016_5.nii.gz', 'image_sphere_WES_017_3.nii.gz', 'image_sphere_WES_018_4.nii.gz', 'image_sphere_WES_018_5.nii.gz', 'image_sphere_WES_018_6.nii.gz', 'image_sphere_WES_019_3.nii.gz', 'image_sphere_WES_019_4.nii.gz', 'image_sphere_WES_019_5.nii.gz', 'image_sphere_WES_021_2.nii.gz', 'image_sphere_WES_021_3.nii.gz', 'image_sphere_WES_021_4.nii.gz', 'image_sphere_WES_023_2.nii.gz', 'image_sphere_WES_023_3.nii.gz', 'image_sphere_WES_023_4.nii.gz', 'image_sphere_WES_024_3.nii.gz', 'image_sphere_WES_024_4.nii.gz', 'image_sphere_WES_024_5.nii.gz']

df=pd.DataFrame()
for image in range(0,10):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=mask_list[image],img_label="MPE")
    dict1.update({'Patient': str(mask_list[image][9:11])})
    dict1.update({'Timepoint': str(mask_list[image][12:13])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_tumour_0_to_9.csv")

df=pd.DataFrame()
for image in range(0,10):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=sphere_list[image],img_label="MPE")
    dict1.update({'Patient':str(sphere_list[image][18:20])})
    dict1.update({'Timepoint': str(sphere_list[image][21:22])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_sphere_0_to_9.csv")

df=pd.DataFrame()
for image in range(10,20):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=mask_list[image],img_label="MPE")
    dict1.update({'Patient': str(mask_list[image][9:11])})
    dict1.update({'Timepoint': str(mask_list[image][12:13])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_tumour_10_to_19.csv")

df=pd.DataFrame()
for image in range(10,20):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=sphere_list[image],img_label="MPE")
    dict1.update({'Patient':str(sphere_list[image][18:20])})
    dict1.update({'Timepoint': str(sphere_list[image][21:22])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_sphere_10_to_19.csv")

df=pd.DataFrame()
for image in range(20,30):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=mask_list[image],img_label="MPE")
    dict1.update({'Patient': str(mask_list[image][9:11])})
    dict1.update({'Timepoint': str(mask_list[image][12:13])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_tumour_20_to_29.csv")

df=pd.DataFrame()
for image in range(20,30):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=sphere_list[image],img_label="MPE")
    dict1.update({'Patient':str(sphere_list[image][18:20])})
    dict1.update({'Timepoint': str(sphere_list[image][21:22])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_sphere_20_to_29.csv")

df=pd.DataFrame()
for image in range(30,40):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=mask_list[image],img_label="MPE")
    dict1.update({'Patient': str(mask_list[image][9:11])})
    dict1.update({'Timepoint': str(mask_list[image][12:13])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_tumour_30_to_39.csv")

df=pd.DataFrame()
for image in range(30,40):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=sphere_list[image],img_label="MPE")
    dict1.update({'Patient':str(sphere_list[image][18:20])})
    dict1.update({'Timepoint': str(sphere_list[image][21:22])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_sphere_30_to_39.csv")

df=pd.DataFrame()
for image in range(40,53):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=mask_list[image],img_label="MPE")
    dict1.update({'Patient': str(mask_list[image][9:11])})
    dict1.update({'Timepoint': str(mask_list[image][12:13])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_tumour_40_to_53.csv")

df=pd.DataFrame()
for image in range(40,53):
    dict1=radiomics_analysis(image_filepath=MPE_list[image], mask_filepath=sphere_list[image],img_label="MPE")
    dict1.update({'Patient':str(sphere_list[image][18:20])})
    dict1.update({'Timepoint': str(sphere_list[image][21:22])})
    df=df.append(dict1,ignore_index=True)
df.to_csv("./df_MPE_sphere_40_to_53.csv")

df1=pd.read_csv("./df_MPE_tumour_0_to_9.csv")
df2=pd.read_csv("./df_MPE_tumour_10_to_19.csv")
df3=pd.read_csv("./df_MPE_tumour_20_to_29.csv")
df4=pd.read_csv("./df_MPE_tumour_30_to_39.csv")
df5=pd.read_csv("./df_MPE_tumour_40_to_53.csv")
df=df1.append(df2)
df=df.append(df3)
df=df.append(df4)
df=df.append(df5)
df.to_csv("./df_MPE_tumours.csv")

df1=pd.read_csv("./df_MPE_sphere_0_to_9.csv")
df2=pd.read_csv("./df_MPE_sphere_10_to_19.csv")
df3=pd.read_csv("./df_MPE_sphere_20_to_29.csv")
df4=pd.read_csv("./df_MPE_sphere_30_to_39.csv")
df5=pd.read_csv("./df_MPE_sphere_40_to_53.csv")
df=df1.append(df2)
df=df.append(df3)
df=df.append(df4)
df=df.append(df5)
df.to_csv("./df_MPE_spheres.csv")
