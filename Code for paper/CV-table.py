import pandas as pd

ADC_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_ADC_CV_full.csv")
DCE_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_DCE_CV_full.csv")
T1w_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_CV_full.csv")
T2w_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_CV_full.csv")
T1w_Z_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_CV_full.csv")
#T2w_Z_Bef_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_CV_full.csv")

ADC_median = ADC_Bef_df.median()
DCE_median = DCE_Bef_df.median()
T1w_median = T1w_Bef_df.median()
T2w_median = T2w_Bef_df.median()
T1w_Z_median = T1w_Z_Bef_df.median()
#T2w_Z_median = T2w_Z_Bef_df.median()


median_df = pd.concat([ADC_median,DCE_median,T1w_median,T2w_median,T1w_Z_median],axis=1)#,T2w_Z_median], axis=1)
median_df = median_df.drop(["Patient"],axis=0)
median_df = median_df.rename(columns={0: 'ADC',1:'DCE',2:'T1w',3:'T2w',4:"T1w Z-norm"})#,5:"T2w Z-norm"})

df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format("06","32"))
features = list(df)
features = features[1:-4]
# print(features)

median_df = median_df.reindex(features)

median_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_median_CV_df.csv")