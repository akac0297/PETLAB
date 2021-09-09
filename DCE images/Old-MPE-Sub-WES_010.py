#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk

import numpy as np

from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


patient_number = "10"
img_path = "/home/alicja/Documents/WES_0" + patient_number + "/IMAGES/"
baseline_img = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_10.nii.gz", sitk.sitkInt16)
image_1 = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_13.nii.gz", sitk.sitkInt16)
image_2 = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_14.nii.gz", sitk.sitkInt16)
image_3 = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_15.nii.gz", sitk.sitkInt16)
image_4 = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_16.nii.gz", sitk.sitkInt16)
image_5 = sitk.ReadImage(img_path + "WES_010_4_20180829_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_17.nii.gz", sitk.sitkInt16)

image_1=sitk.Resample(image_1, baseline_img)
image_2=sitk.Resample(image_2, baseline_img)
image_3=sitk.Resample(image_3, baseline_img)
image_4=sitk.Resample(image_4, baseline_img)
image_5=sitk.Resample(image_5, baseline_img)

new_img_1=image_1-baseline_img
new_img_2=image_2-baseline_img
new_img_3=image_3-baseline_img
new_img_4=image_4-baseline_img
new_img_5=image_5-baseline_img


# In[ ]:


im_arr = [image_2, image_3, image_4, image_5]
# create new array, deepcopy of image_1
import copy
max_arr = copy.deepcopy(image_1)
#find max of all locations of 5 images
print("width= " + str(image_1.GetWidth()))
print("height= " + str(image_1.GetHeight()))
print("depth= " + str(image_1.GetDepth()))

for image in im_arr:
    for w in range(image_1.GetWidth()):
        for h in range(image_1.GetHeight()):
            for d in range(image_1.GetDepth()):
                if max_arr[w,h,d] < image[w,h,d]:
                    max_arr[w,h,d] = image[w,h,d]
        print("column: " + str(w) + " completed")


# In[ ]:


sitk.WriteImage(max_arr, "MPE_sub_WES_010_4.nii.gz")
vis = ImageVisualiser(max_arr, window=[-100, 300], figure_size_in=5)
fig = vis.show()


# In[ ]:


patient_number = "10"
img_path = "/home/alicja/Documents/WES_0" + patient_number + "/IMAGES/"
baseline_img = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_9.nii.gz", sitk.sitkInt16)
image_1 = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_12.nii.gz", sitk.sitkInt16)
image_2 = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_13.nii.gz", sitk.sitkInt16)
image_3 = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_14.nii.gz", sitk.sitkInt16)
image_4 = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_15.nii.gz", sitk.sitkInt16)
image_5 = sitk.ReadImage(img_path + "WES_010_5_20181010_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_16.nii.gz", sitk.sitkInt16)

image_1=sitk.Resample(image_1, baseline_img)
image_2=sitk.Resample(image_2, baseline_img)
image_3=sitk.Resample(image_3, baseline_img)
image_4=sitk.Resample(image_4, baseline_img)
image_5=sitk.Resample(image_5, baseline_img)

new_img_1=image_1-baseline_img
new_img_2=image_2-baseline_img
new_img_3=image_3-baseline_img
new_img_4=image_4-baseline_img
new_img_5=image_5-baseline_img


# In[ ]:


im_arr = [image_2, image_3, image_4, image_5]
# create new array, deepcopy of image_1
import copy
max_arr = copy.deepcopy(image_1)
#find max of all locations of 5 images
print("width= " + str(image_1.GetWidth()))
print("height= " + str(image_1.GetHeight()))
print("depth= " + str(image_1.GetDepth()))

for image in im_arr:
    for w in range(image_1.GetWidth()):
        for h in range(image_1.GetHeight()):
            for d in range(image_1.GetDepth()):
                if max_arr[w,h,d] < image[w,h,d]:
                    max_arr[w,h,d] = image[w,h,d]
        print("column: " + str(w) + " completed")


# In[ ]:


sitk.WriteImage(max_arr, "MPE_sub_WES_010_5.nii.gz")
vis = ImageVisualiser(max_arr, window=[-100, 300], figure_size_in=5)
fig = vis.show()


# In[ ]:


patient_number = "10"
img_path = "/home/alicja/Documents/WES_0" + patient_number + "/IMAGES/"
baseline_img = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_9.nii.gz", sitk.sitkInt16)
image_1 = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_12.nii.gz", sitk.sitkInt16)
image_2 = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_13.nii.gz", sitk.sitkInt16)
image_3 = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_14.nii.gz", sitk.sitkInt16)
image_4 = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_15.nii.gz", sitk.sitkInt16)
image_5 = sitk.ReadImage(img_path + "WES_010_6_20190301_MR_T1_FL3D_TRA_DYNAVIEWS_BAYER_FL3D1_T1_FL3D_TRA_DYNAVIEWS_BAYER_16.nii.gz", sitk.sitkInt16)

image_1=sitk.Resample(image_1, baseline_img)
image_2=sitk.Resample(image_2, baseline_img)
image_3=sitk.Resample(image_3, baseline_img)
image_4=sitk.Resample(image_4, baseline_img)
image_5=sitk.Resample(image_5, baseline_img)

new_img_1=image_1-baseline_img
new_img_2=image_2-baseline_img
new_img_3=image_3-baseline_img
new_img_4=image_4-baseline_img
new_img_5=image_5-baseline_img


# In[ ]:


im_arr = [image_2, image_3, image_4, image_5]
# create new array, deepcopy of image_1
import copy
max_arr = copy.deepcopy(image_1)
#find max of all locations of 5 images
print("width= " + str(image_1.GetWidth()))
print("height= " + str(image_1.GetHeight()))
print("depth= " + str(image_1.GetDepth()))

for image in im_arr:
    for w in range(image_1.GetWidth()):
        for h in range(image_1.GetHeight()):
            for d in range(image_1.GetDepth()):
                if max_arr[w,h,d] < image[w,h,d]:
                    max_arr[w,h,d] = image[w,h,d]
        print("column: " + str(w) + " completed")


# In[ ]:


sitk.WriteImage(max_arr, "MPE_sub_WES_010_6.nii.gz")
vis = ImageVisualiser(max_arr, window=[-100, 300], figure_size_in=5)
fig = vis.show()


# In[ ]:





# In[ ]:




