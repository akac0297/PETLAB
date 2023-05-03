import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib
from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com
from platipy.imaging.registration.utils import smooth_and_resample 

mask_path_MRI="/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/PETLAB Breast Masks/Contralateral-breasts/"

def getMRIMaskFilepaths(mask_path_MRI,patient_no):
    # This is for whole breast radiomics - contralateral breast
    Bef_Mask=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_1_T2W_contralateral.nii.gz")
    Dur_Mask=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_2_T2W_contralateral.nii.gz")
    Post_Mask=str(f"{mask_path_MRI}WES_0{patient_no}_TIMEPOINT_3_T2W_contralateral.nii.gz")

    return(Bef_Mask, Dur_Mask, Post_Mask)

def erodeMask(mask_filepath, patient_no):
    mask = sitk.ReadImage(mask_filepath, sitk.sitkUInt8)
    shrink = [int(10/s) for s in mask.GetSpacing()]

    vis = ImageVisualiser(mask, cut=get_com(mask))
    fig = vis.show()
    fig.savefig(f"./original_test_mask_patient_0{patient_no}.jpeg", dpi=500)

    mask = sitk.BinaryErode(mask,shrink)

    vis = ImageVisualiser(mask, cut=get_com(mask))
    fig = vis.show()
    fig.savefig(f"./eroded_test_mask_patient_0{patient_no}.jpeg", dpi=500)

    return(mask)

# patient_no = '19'
# Bef_Mask, Dur_Mask, Post_Mask = getMRIMaskFilepaths(mask_path_MRI,patient_no)
# mask = erodeMask(Bef_Mask,patient_no)
# sitk.WriteImage(mask,f'./eroded_test_mask_patient_0{patient_no}.nii.gz')

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

patient_no = '15'
Bef_Mask, Dur_Mask, Post_Mask = getMRIMaskFilepaths(mask_path_MRI,patient_no)
mask_path_MRI="/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/"
Post_Mask=str(f"{mask_path_MRI}/WES_0{patient_no}/STRUCTURES/WES_0{patient_no}_TIMEPOINT_3_GTV.nii.gz")
Bef_Pre,Dur_Pre,Post_Pre=getT1wImgFilepaths(path,patient_no)
Bef_ME,Dur_ME,Post_ME=getMEImgFilepaths(path,patient_no)
Bef_T2w_SPAIR, Dur_T2w_SPAIR, Post_T2w_SPAIR=getT2wImgFilepaths(path,patient_no)
Bef_ADC,Dur_ADC,Post_ADC=getADCImgFilepaths(path,patient_no)
Post_Mask_img = sitk.ReadImage(Post_Mask)
Post_Pre_img = sitk.ReadImage(Post_Pre)
Post_ME_img = sitk.ReadImage(Post_ME)
Post_T2w_SPAIR_img = sitk.ReadImage(Post_T2w_SPAIR)
Post_ADC_img = sitk.ReadImage(Post_ADC)

Post_Pre_img=smooth_and_resample(Post_Pre_img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)
Post_ME_img=smooth_and_resample(Post_ME_img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)
Post_T2w_SPAIR_img=smooth_and_resample(Post_T2w_SPAIR_img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)
Post_ADC_img=smooth_and_resample(Post_ADC_img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)
Post_Mask_img=smooth_and_resample(Post_Mask_img,isotropic_voxel_size_mm=0.5,interpolator=sitk.sitkLinear)

Post_Mask_arr = sitk.GetArrayFromImage(Post_Mask_img)
print(np.max(Post_Mask_arr))

sitk.WriteImage(Post_Pre_img,'./test_Post_Pre_img_patient_15.nii.gz')
sitk.WriteImage(Post_ME_img,'./test_Post_ME_img_patient_15.nii.gz')
sitk.WriteImage(Post_T2w_SPAIR_img,'./test_Post_T2w_SPAIR_img_patient_15.nii.gz')
sitk.WriteImage(Post_ADC_img,'./test_Post_ADC_img_patient_15.nii.gz')
sitk.WriteImage(Post_Mask_img,'./test_Post_Mask_img_patient_15.nii.gz')