import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# icc_repr_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df_75.csv")
# # print(icc_repr_df)

# features = ['Correlation','Idmn','Idn','Imc1','Imc2','MCC','GrayLevelNonUniformity','ZonePercentage','LongRunEmphasis','RunLengthNonUniformity','RunLengthNonUniformityNormalized','RunPercentage','ShortRunEmphasis','Coarseness','DependenceNonUniformity','DependenceNonUniformityNormalized','DependenceVariance','SmallDependenceEmphasis','SmallDependenceLowGrayLevelEmphasis']

# icc_repr_df=icc_repr_df.loc[icc_repr_df['Unnamed: 0'].isin(features)].reset_index(drop=True)
# icc_repr_df.rename(columns={'Unnamed: 0':'Feature'},inplace=True)
# print(icc_repr_df)

# adc_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_ADC_CV_filtered.csv")
# adc_df=adc_df.drop('Patient', axis=1)
# print(adc_df)

# DCE_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_DCE_CV_filtered.csv")
# DCE_df=DCE_df.drop('Patient', axis=1)
# print(DCE_df)

# T1w_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_CV_filtered.csv")
# T1w_df=T1w_df.drop('Patient', axis=1)
# print(T1w_df)

# T2w_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_CV_filtered.csv")
# T2w_df=T2w_df.drop('Patient', axis=1)
# print(T2w_df)

# new_adc_df=pd.DataFrame()
# df=pd.DataFrame()
# features=list(adc_df)
# for i in range(len(features)):
#     df["CV"]=adc_df.iloc[:,i]
#     df["Feature"]=features[i]
#     new_adc_df = pd.concat([new_adc_df,df],axis=0)
# new_adc_df["Sequence"]="ADC"
# print(new_adc_df)

# new_DCE_df=pd.DataFrame()
# df=pd.DataFrame()
# features=list(DCE_df)
# for i in range(len(features)):
#     df["CV"]=DCE_df.iloc[:,i]
#     df["Feature"]=features[i]
#     new_DCE_df = pd.concat([new_DCE_df,df],axis=0)
# new_DCE_df["Sequence"]="DCE"
# print(new_DCE_df)

# new_T1w_df=pd.DataFrame()
# df=pd.DataFrame()
# features=list(T1w_df)
# for i in range(len(features)):
#     df["CV"]=T1w_df.iloc[:,i]
#     df["Feature"]=features[i]
#     new_T1w_df = pd.concat([new_T1w_df,df],axis=0)
# new_T1w_df["Sequence"]="T1w"
# print(new_DCE_df)

# new_T2w_df=pd.DataFrame()
# df=pd.DataFrame()
# features=list(T2w_df)
# for i in range(len(features)):
#     df["CV"]=T2w_df.iloc[:,i]
#     df["Feature"]=features[i]
#     new_T2w_df = pd.concat([new_T2w_df,df],axis=0)
# new_T2w_df["Sequence"]="T2w"
# print(new_T2w_df)

# final_df=pd.concat([new_adc_df,new_DCE_df,new_T1w_df,new_T2w_df],axis=0)
# print(final_df)

# sns.set_style("whitegrid")

# plt.figure()
# p1=sns.stripplot(x="CV",y="Feature",data=new_adc_df,edgecolor="w",linewidth=0.3)
# p1.set_xlim(0,100)
# plt.tight_layout()
# plt.grid(axis='both')
# plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_CV_stripplot.png")

# plt.figure()
# p2=sns.stripplot(x="CV",y="Feature",data=new_DCE_df,edgecolor="w",linewidth=0.3)
# p2.set_xlim(0,100)
# plt.tight_layout()
# plt.grid(axis='both')
# plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_CV_stripplot.png")

# sns.set(rc={'figure.figsize':(8,5)})
# sns.set_style("whitegrid")
# plt.figure()
# p3=sns.stripplot(x="CV",y="Feature",data=new_T1w_df,edgecolor="w",linewidth=0.3)
# p3.set_xlim(0,100)
# plt.tight_layout()
# plt.grid(axis='both')
# plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T1w_CV_stripplot.png")

# plt.figure()
# p4=sns.stripplot(x="CV",y="Feature",data=new_T2w_df,edgecolor="w",linewidth=0.3)
# p4.set_xlim(0,100)
# plt.tight_layout()
# plt.grid(axis='both')
# plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2w_CV_stripplot.png")

# # plt.figure()
# # p5=sns.catplot(data=final_df,x="CV",y="Feature",col="Sequence")
# # plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Test_fig_combined.png")

"""
Stats across timepoints
"""

icc_repr_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_icc_df_75.csv")
# print(icc_repr_df)
features=list(icc_repr_df["Unnamed: 0"].values)
features.remove("Uniformity")
features.remove("10Percentile")

icc_repr_df=icc_repr_df.loc[icc_repr_df['Unnamed: 0'].isin(features)].reset_index(drop=True)
icc_repr_df.rename(columns={'Unnamed: 0':'Feature'},inplace=True)
print(icc_repr_df)

adc_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_ADC_CV_filtered.csv")
adc_df=adc_df.drop('Patient', axis=1)
print(adc_df)

DCE_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_DCE_CV_filtered.csv")
DCE_df=DCE_df.drop('Patient', axis=1)
print(DCE_df)

T1w_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_T1w_CV_filtered.csv")
T1w_df=T1w_df.drop('Patient', axis=1)
print(T1w_df)

T2w_df = pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_T2w_CV_filtered.csv")
T2w_df=T2w_df.drop('Patient', axis=1)
print(T2w_df)

new_adc_df=pd.DataFrame()
df=pd.DataFrame()
features=list(adc_df)
for i in range(len(features)):
    df["CV"]=adc_df.iloc[:,i]
    df["Feature"]=features[i]
    new_adc_df = pd.concat([new_adc_df,df],axis=0)
new_adc_df["Sequence"]="ADC"
print(new_adc_df)

new_DCE_df=pd.DataFrame()
df=pd.DataFrame()
features=list(DCE_df)
for i in range(len(features)):
    df["CV"]=DCE_df.iloc[:,i]
    df["Feature"]=features[i]
    new_DCE_df = pd.concat([new_DCE_df,df],axis=0)
new_DCE_df["Sequence"]="DCE"
print(new_DCE_df)

new_T1w_df=pd.DataFrame()
df=pd.DataFrame()
features=list(T1w_df)
for i in range(len(features)):
    df["CV"]=T1w_df.iloc[:,i]
    df["Feature"]=features[i]
    new_T1w_df = pd.concat([new_T1w_df,df],axis=0)
new_T1w_df["Sequence"]="T1w"
print(new_DCE_df)

new_T2w_df=pd.DataFrame()
df=pd.DataFrame()
features=list(T2w_df)
for i in range(len(features)):
    df["CV"]=T2w_df.iloc[:,i]
    df["Feature"]=features[i]
    new_T2w_df = pd.concat([new_T2w_df,df],axis=0)
new_T2w_df["Sequence"]="T2w"
print(new_T2w_df)

final_df=pd.concat([new_adc_df,new_DCE_df,new_T1w_df,new_T2w_df],axis=0)
print(final_df)

sns.set_style("whitegrid")

plt.figure()
p2=sns.stripplot(x="CV",y="Feature",data=new_DCE_df,edgecolor="w",linewidth=0.3)
p2.set_xlim(0,100)
plt.tight_layout()
plt.grid(axis='both')
plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_CV_stripplot_BC64.png")

plt.figure()
p4=sns.stripplot(x="CV",y="Feature",data=new_T2w_df,edgecolor="w",linewidth=0.3)
p4.set_xlim(0,100)
plt.tight_layout()
plt.grid(axis='both')
plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2w_CV_stripplot_BC64.png")

sns.set(rc={'figure.figsize':(8,5)})
sns.set_style("whitegrid")
plt.figure()
p3=sns.stripplot(x="CV",y="Feature",data=new_T1w_df,edgecolor="w",linewidth=0.3)
p3.set_xlim(0,100)
plt.tight_layout()
plt.grid(axis='both')
plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T1w_CV_stripplot_BC64.png")

plt.figure()
p1=sns.stripplot(x="CV",y="Feature",data=new_adc_df,edgecolor="w",linewidth=0.3)
p1.set_xlim(0,100)
plt.tight_layout()
plt.grid(axis='both')
plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_CV_stripplot_BC64.png")