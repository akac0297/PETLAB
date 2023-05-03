import pandas as pd

# """ Within effects """

# ADC_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_ADC_CV_full.csv")
# DCE_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_DCE_CV_full.csv")
# T1w_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_CV_full.csv")
# T2w_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_CV_full.csv")
# T1w_Z_effect_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_effect_CV_full.csv")
# T2w_Z_effect_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_effect_CV_full.csv")

# ADC_median = ADC_Bef_df.median()
# DCE_median = DCE_Bef_df.median()
# T1w_median = T1w_Bef_df.median()
# T2w_median = T2w_Bef_df.median()
# T1w_Z_effect_median = T1w_Z_effect_Bef_df.median()
# T2w_Z_effect_median = T2w_Z_effect_Bef_df.median()


# median_df = pd.concat([ADC_median,DCE_median,T1w_median,T2w_median,T1w_Z_effect_median,T2w_Z_effect_median], axis=1)
# median_df = median_df.drop(["Patient"],axis=0)
# median_df = median_df.rename(columns={0: 'ADC',1:'DCE',2:'T1w',3:'T2w',4:"T1w Z-norm effect",5:"T2w Z-norm effect"})

# df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format("06","32"))
# features = list(df)
# features = features[1:-4]
# # print(features)

# median_df = median_df.reindex(features)

# median_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_median_CV_df.csv")

""" Across timepoints """

binCount = '64' #'32'

ADC_df = pd.read_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{binCount}_contralateral_ADC_CV_full.csv")
DCE_df = pd.read_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{binCount}_contralateral_DCE_CV_full.csv")
T1w_df = pd.read_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{binCount}_contralateral_T1w_CV_full.csv")
T2w_df = pd.read_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{binCount}_contralateral_T2w_CV_full.csv")

ADC_median = ADC_df.median()
DCE_median = DCE_df.median()
T1w_median = T1w_df.median()
T2w_median = T2w_df.median()

median_df = pd.concat([ADC_median,DCE_median,T1w_median,T2w_median], axis=1)
median_df = median_df.drop(["Patient"],axis=0)
median_df = median_df.rename(columns={0: 'ADC',1:'DCE',2:'T1w',3:'T2w'})

patient_no = '06'
df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format(patient_no,binCount))
features = list(df)
features = features[1:-4]
# print(features)

median_df = median_df.reindex(features)

median_df.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{binCount}_contralateral_median_CV_df.csv")