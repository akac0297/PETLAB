import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
# import numpy as np

# icc_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df_new.csv")
# icc_df=icc_df.set_index("Unnamed: 0")

# df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format("06","32"))
# features = list(df)
# features = features[1:-4]

# icc_df = icc_df.reindex(features)

# df_1 = icc_df.iloc[:18,:]
# df_2 = icc_df.iloc[18:32,:]
# df_3 = icc_df.iloc[32:56,:]
# df_4 = icc_df.iloc[56:72,:]
# df_5 = icc_df.iloc[72:88,:]
# df_6 = icc_df.iloc[72:93,:]
# df_7 = icc_df.iloc[93:,:]

# def plotHeatmap(icc_df, matrix_type):
#     # make the values in dataset discrete
#     df_q = pd.DataFrame()
#     for col in icc_df:
#         df_q[col] = pd.cut(abs(icc_df[col].astype(float)), [0,0.5,0.75,0.9,1.01],right=False,labels=[0,1,2,3])
#     df_q = df_q[df_q.columns].astype(float)

#     # plot heatmap
#     sns.set(font_scale=1)
#     cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=4)
#     grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
#     bounds = [0, 0.5, 0.75, 0.9, 1]
#     my_norm = BoundaryNorm(bounds, ncolors=5)
#     # Create two appropriately sized subplots
#     _, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
#     # ax = sns.heatmap(icc_df, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap), linewidths=.5, linecolor='lightgray', cbar_kws={'orientation': 'vertical'})
#     ax = sns.heatmap(icc_df,
#             ax=ax,
#             cmap=ListedColormap(cmap),
#             norm=my_norm,
#             cbar_ax=cbar_ax,
#             linewidths=0, 
#             linecolor='lightgray',
#             xticklabels=True, 
#             yticklabels=True,
#             cbar_kws={'orientation': 'vertical'})

#     # Customize tick marks and positions
#     cbar_ax.set_yticklabels(['0','0.5', '0.75', '0.9', '1.0'])
#     cbar_ax.yaxis.set_ticks([ 0, 0.5, 0.75, 0.9, 1.0])

#     # Rotate tick labels
#     _, labels = plt.xticks()
#     plt.setp(labels, rotation=0)
#     _, labels = plt.yticks()
#     plt.setp(labels, rotation=0)
#     ax.set(ylabel=None)
#     plt.savefig(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_icc_heatmap_{matrix_type}_new.png",bbox_inches='tight')

# plotHeatmap(df_1,"1st_order")
# plotHeatmap(df_2,"shape")
# plotHeatmap(df_3,"GLCM")
# plotHeatmap(df_4,"GLRLM")
# plotHeatmap(df_5,"GLSZM")
# plotHeatmap(df_6,"NGTDM")
# plotHeatmap(df_7,"GLDM")

icc_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_contralateral_icc_df_new.csv")
icc_df=icc_df.set_index("Unnamed: 0")

df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format("06","32"))
features = list(df)
features = features[1:-4]

icc_df = icc_df.reindex(features)

df_1 = icc_df.iloc[:18,:]
df_2 = icc_df.iloc[18:32,:]
df_3 = icc_df.iloc[32:56,:]
df_4 = icc_df.iloc[56:72,:]
df_5 = icc_df.iloc[72:88,:]
df_6 = icc_df.iloc[72:93,:]
df_7 = icc_df.iloc[93:,:]

def plotHeatmap(icc_df, matrix_type):
    # make the values in dataset discrete
    df_q = pd.DataFrame()
    for col in icc_df:
        df_q[col] = pd.cut(abs(icc_df[col].astype(float)), [0,0.5,0.75,0.9,1.01],right=False,labels=[0,1,2,3])
    df_q = df_q[df_q.columns].astype(float)

    # plot heatmap
    sns.set(font_scale=1)
    cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=4)
    grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
    bounds = [0, 0.5, 0.75, 0.9, 1]
    my_norm = BoundaryNorm(bounds, ncolors=5)
    # Create two appropriately sized subplots
    _, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
    # ax = sns.heatmap(icc_df, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap), linewidths=.5, linecolor='lightgray', cbar_kws={'orientation': 'vertical'})
    ax = sns.heatmap(icc_df,
            ax=ax,
            cmap=ListedColormap(cmap),
            norm=my_norm,
            cbar_ax=cbar_ax,
            linewidths=0, 
            linecolor='lightgray',
            xticklabels=True, 
            yticklabels=True,
            cbar_kws={'orientation': 'vertical'})

    # Customize tick marks and positions
    cbar_ax.set_yticklabels(['0','0.5', '0.75', '0.9', '1.0'])
    cbar_ax.yaxis.set_ticks([ 0, 0.5, 0.75, 0.9, 1.0])

    # Rotate tick labels
    _, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    _, labels = plt.yticks()
    plt.setp(labels, rotation=0)
    ax.set(ylabel=None)
    plt.savefig(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount64_icc_heatmap_{matrix_type}_new.png",bbox_inches='tight')

plotHeatmap(df_1,"1st_order")
plotHeatmap(df_2,"shape")
plotHeatmap(df_3,"GLCM")
plotHeatmap(df_4,"GLRLM")
plotHeatmap(df_5,"GLSZM")
plotHeatmap(df_6,"NGTDM")
plotHeatmap(df_7,"GLDM")