import pandas as pd
import numpy as np

df = pd.DataFrame()

for i in range(3):
    dict1 = {"Mean": 0.7, "Std": 0.01, "Max": 0.8}
    dict2 = {"Min": 0.6, "Skewness": 0.4, "Max": 0.9}

    df1=pd.Series(dict1)
    df2=pd.Series(dict2)
    df_new=pd.concat([df1,df2],axis=0)
    df_new["Patient"] = i
    df=df.append(df_new,ignore_index=True)

print(df)