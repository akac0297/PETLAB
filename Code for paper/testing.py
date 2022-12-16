"""
The final radiomics shape features for the same patient/timepoint are different for the different image series. But I think they should be the same. 
One way to get this to work is to resample the different images to one same image. But if the tumour mask covers the same area for all the sequences, then 
there is no need to resample?

- I have resampled the mask to the different sequences and save this new mask somewhere, then
- opened up the new masks with the appropriate images in Slicer and double checked that they overlap well and it seems pretty good.
- I also resampled the ADC and DCE images to the T2w SPAIR and checked the overlay of the normal mask on these resampled images. This seemed to work well
and also, the resampled images aligned pretty closely over each other
- Could there be an issue with the isotropic voxel resampling in the radiomics code?

"""
import SimpleITK as sitk
import numpy as np

path="/home/alicja/PET_LAB_PROCESSED/WES_014/IMAGES/"

ADC_img = sitk.ReadImage(path+"WES_014_TIMEPOINT_1_MRI_DWI_ADC.nii.gz")
DCE_img = sitk.ReadImage(path+"WES_014_TIMEPOINT_1_MRI_T1W_DCE_ACQ_0.nii.gz")
T2_SPAIR_img = sitk.ReadImage(path+"WES_014_TIMEPOINT_1_MRI_T2W_SPAIR.nii.gz")

mask = sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_1_GTV.nii.gz")

ADC_mask = sitk.Resample(mask, ADC_img, sitk.Transform(),sitk.sitkNearestNeighbor)
DCE_mask = sitk.Resample(mask, DCE_img, sitk.Transform(),sitk.sitkNearestNeighbor)

sitk.WriteImage(ADC_mask, "/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_1_GTV_resampled_to_ADC.nii.gz")
sitk.WriteImage(DCE_mask, "/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_1_GTV_resampled_to_DCE.nii.gz")

ADC_resampled = sitk.Resample(ADC_img, T2_SPAIR_img, sitk.Transform(), sitk.sitkNearestNeighbor)
DCE_resampled = sitk.Resample(DCE_img, T2_SPAIR_img, sitk.Transform(), sitk.sitkNearestNeighbor)

sitk.WriteImage(ADC_resampled, path+"WES_014_TIMEPOINT_1_MRI_DWI_ADC_resampled_to_T2_SPAIR.nii.gz")
sitk.WriteImage(DCE_resampled, path+"WES_014_TIMEPOINT_1_MRI_DWI_DCE_ACQ_0_resampled_to_T2_SPAIR.nii.gz")

print("Size of ADC image:", ADC_img.GetSize())
print("Spacing of ADC image:", ADC_img.GetSpacing())

print("Size of DCE image:", DCE_img.GetSize())
print("Spacing of DCE image:", DCE_img.GetSpacing())

print("Size of resampled ADC image:", ADC_resampled.GetSize())
print("Spacing of resampled ADC image:", ADC_resampled.GetSpacing())

print("Size of resampled DCE image:", DCE_resampled.GetSize())
print("Spacing of resampled DCE image:", DCE_resampled.GetSpacing())

print("Size of T2_SPAIR image:", T2_SPAIR_img.GetSize())
print("Spacing of T2_SPAIR image:", T2_SPAIR_img.GetSpacing())

print("Size of mask:", mask.GetSize())
print("Spacing of mask:", mask.GetSpacing())

# print("ADC",[160*2,124*2,48*6])
# print("DCE",[336*1.25,256*1.25,192*1.4])
# print("T2w",[504*0.833333,384*0.833333,60*4.2])