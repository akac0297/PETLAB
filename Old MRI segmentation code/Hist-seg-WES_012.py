#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import modules
import SimpleITK as sitk

from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[5]:


from platipy.imaging.visualisation.tools import ImageVisualiser

from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration,
    transform_propagation,
    apply_field
)


# In[6]:


WES_012_4_B50T=sitk.ReadImage("/home/alicja/Documents/WES_012/IMAGES/WES_012_4_20180912_MR_EP2D_DIFF_TRA_SPAIR_ZOOMIT_EZ_B50T_EP2D_DIFF_TRA_SPAIR_ZOOMIT_TRACEW_DFC_MIX_7.nii.gz")
WES_012_4_B800T=sitk.ReadImage("/home/alicja/Documents/WES_012/IMAGES/WES_012_4_20180912_MR_EP2D_DIFF_TRA_SPAIR_ZOOMIT_EZ_B800T_EP2D_DIFF_TRA_SPAIR_ZOOMIT_TRACEW_DFC_MIX_7.nii.gz")
WES_012_4_T2w=sitk.ReadImage("/home/alicja/Documents/WES_012/IMAGES/WES_012_4_20180912_MR_T2_TSE_TRA_SPAIR_TSE2D1_11_T2_TSE_TRA_SPAIR_3.nii.gz")
WES_012_4_MPE=sitk.ReadImage("MPE_sub_WES_012_4.nii.gz")


# In[ ]:




