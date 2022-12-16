import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

icc_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df.csv")
icc_df=icc_df.set_index("Unnamed: 0")

df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format("06","32"))
features = list(df)
features = features[1:-4]

icc_df = icc_df.reindex(features)

# make the values in dataset discrete
# the values will be cut into 3 discrete values: 0,1,2
df_q = pd.DataFrame()
for col in icc_df:
    df_q[col] = pd.cut(abs(icc_df[col].astype(float)), [0,0.5,0.75,0.9,1.01],right=False,labels=[0,1,2,3])

print(df_q)
df_q = df_q[df_q.columns].astype(float)
# plot it
sns.heatmap(df_q, annot=True)
plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_icc_heatmap.png")