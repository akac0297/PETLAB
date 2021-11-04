import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mrmr import mrmr_classif
from sklearn import metrics

"""
- need to import DCE and PET datasets and make sure they cover the same patients

Two cases:
- MRMR predicting PET-CT volume (10 features)

Gaussian Naive Bayes using the combined model to predict PET-CT volume

"""

def obtainX():
    DCE_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE-MRI_radiomics_features_shortened.csv")
    indices=np.linspace(156,191,num=36)
    DCE_df=DCE_df.drop(indices)
    DCE_df=DCE_df.drop(['Patient', 'image label','Unnamed: 0'], axis=1)
    columns=list(DCE_df)
    image_labels=['Bef Pre','Bef Post1', 'Bef Post2', 'Bef Post3', 'Bef Post4', 'Bef Post 5','Dur Pre','Dur Post1', 'Dur Post2', 'Dur Post3', 'Dur Post4', 'Dur Post 5']

    new_labels=[]
    for feature in columns:
        for label in image_labels:
            new_labels.append(feature + " " + label)

    data=DCE_df.values
    X = data.reshape(-1,120) #each individual DCE image produces 10 features, and there are 12 images per patient, so 120 radiomics features per patient in total.
    DCE=X
    X_df=pd.DataFrame(X,columns=[new_labels])
    DCE_df=X_df
    PET_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_radiomics_features_shortened_new.csv")
    PET_df=PET_df.drop(['Patient', 'image label','Unnamed: 0'], axis=1)
    columns=list(PET_df)
    image_labels=['Bef PET','Dur PET']
    new_labels=[]
    for feature in columns:
        for label in image_labels:
            new_labels.append(feature + " " + label)

    data=PET_df.values
    X = data.reshape(-1,16) #each individual PET image produces 8 features, and there are 2 images per patient, so 16 radiomics features per patient in total.
    PET=X
    X_df=pd.DataFrame(X,columns=[new_labels])
    PET_df=X_df

    X=np.concatenate((DCE,PET),axis=1)
    X_df=pd.concat([DCE_df,PET_df], axis=1)
    return(X,X_df)

PET_vols = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]).reshape(-1,1)

X,X_df=obtainX()
print(X)
print(X_df)

def getXRelevant(X,y,X_df):
    relevant_features = mrmr_classif(X,y,K=8)
    print(relevant_features)
    X_relevant = X[:,relevant_features]
    X_relevant_df=X_df.iloc[:,relevant_features]
    return X_relevant_df,X_relevant

X_relevant_df,X_relevant=getXRelevant(X,PET_vols,X_df)

X_relevant_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_and_PET_relevant_df_PET-CT_vols.csv")
X_relevant_df["PET-CT Tumour volume change"]=PET_vols

"""
Modelling: Use LR and GNB
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler 

LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
GNB = GaussianNB()

cross_validator = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)

PET_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_and_PET_relevant_df_PET-CT_vols.csv",index_col=0)
PET_volumes=np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

kf = KFold(n_splits=5)

LRaccuracies=[]
GNBaccuracies=[]
LR_AUCs=[]
GNB_AUCs=[]

y_pred_LRs=[]
y_pred_GNBs=[]

LR_y_pred_probas=np.empty((13,2),float)
GNB_y_pred_probas=np.empty((13,2),float)
LR_fprs=[]
LR_tprs=[]
GNB_fprs=[]
GNB_tprs=[]

LR_sensitivities=[]
LR_specificities=[]
GNB_sensitivities=[]
GNB_specificities=[]

test_splits=[[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]
train_splits=[[i for i in range(13) if i not in j] for j in test_splits]

for i in range(13):
    train_indices = list(range(13))
    train_indices.remove(i)
    df_train = PET_relevant_df.iloc[train_indices]
    PET_vol_train=PET_volumes[train_indices]
    df_test = PET_relevant_df.iloc[i]
    PET_vol_test=PET_volumes[i]

    ss = StandardScaler()
    ss.fit(df_train)
    X_train = ss.transform(df_train)
    X_test = ss.transform([df_test])
    y_train = np.array([0,1,1,0,0,1,1,1,0,0,1,0,1])
    y_test=np.array(y_train[i])
    y_train=list(y_train)
    y_train.pop(i)
    y_train=np.array(y_train)
    print("y train:",y_train)
    LR.fit(X_train, y_train)
    LR.predict_proba(X_test)

    LR.fit(X_train,PET_vol_train)
    GNB.fit(X_train,PET_vol_train)

    #for LR
    importance=LR.coef_
    #print("Importance for LR PET: ",importance)

    y_pred_LR = LR.predict(X_test)
    #print("LR predicted PET vols: ",y_pred_LR)
    #print("Correct PET vols: ",PET_vol_test)
    #print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_LR)))

    #for gausian NB
    y_pred_GNB = GNB.predict(X_test)
    #print("GNB predicted PET vols: ",y_pred_GNB)
    #print("Correct PET vols: ",PET_vol_test)
    #print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_GNB)))

    LR_y_pred_proba = LR.predict_proba(X_test)
    GNB_y_pred_proba = GNB.predict_proba(X_test)

    y_pred_LRs.append(y_pred_LR)
    y_pred_GNBs.append(y_pred_GNB)

    LR_y_pred_probas[i,:]=LR_y_pred_proba
    GNB_y_pred_probas[i,:]=GNB_y_pred_proba

print(y_pred_LRs)
print(y_pred_GNBs)
print(PET_volumes)

print(LR_y_pred_probas)
print(GNB_y_pred_probas)

print(f"Number of LR mislabeled points out of a total {PET_volumes.shape[0]} points : {np.sum(PET_volumes != np.array(y_pred_LRs).reshape(1,-1)[0])}")
print(f"Number of GNB mislabeled points out of a total {PET_volumes.shape[0]} points : {np.sum(PET_volumes != np.array(y_pred_GNBs).reshape(1,-1)[0])}")

"""
Compute and plot an ROC curve
"""

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(PET_volumes, LR_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for multimodality Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for multimodality LR:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_combined_LR.png')

fpr, tpr, _ = roc_curve(PET_volumes, GNB_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for multimodality Gaussian Naive Bayes")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for multimodality GNB:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_combined_GNB.png')

"""
Accuracy, sensitivity, specificity
"""

from sklearn.metrics import confusion_matrix

#Confusion matrix, Accuracy, sensitivity and specificity

cm1_LR = confusion_matrix(PET_volumes,y_pred_LRs)
print('Confusion Matrix LR: \n', cm1_LR)

total1=sum(sum(cm1_LR))
##from confusion matrix calculate accuracy
accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
print ('Accuracy LR : ', accuracy1)

sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
print('Sensitivity LR : ', sensitivity1 )

specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
print('Specificity LR : ', specificity1)

cm1_GNB = confusion_matrix(PET_volumes,y_pred_GNBs)
print('Confusion Matrix GNB: \n', cm1_GNB)

total1=sum(sum(cm1_GNB))
##from confusion matrix calculate accuracy
accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
print ('Accuracy GNB: ', accuracy1)

sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
print('Sensitivity GNB: ', sensitivity1 )

specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
print('Specificity GNB: ', specificity1)

# i=0
# for train_index, test_index in zip(train_splits,test_splits):
#     print(test_index)
#     length=len(test_splits[i])
#     print("length:",length)
#     train_index=[int(i) for i in train_index]
#     test_index=[int(i) for i in test_index]
#     PET_train, PET_test = PET_relevant_df.iloc[train_index].values, PET_relevant_df.iloc[test_index].values
#     PET_vol_train, PET_vol_test = PET_volumes[train_index], PET_volumes[test_index]

#     ss = StandardScaler()
#     ss.fit(PET_train)
#     X_train = ss.transform(PET_train)
#     X_test = ss.transform(PET_test)
#     y_train = np.array([0,1,1,0,0,1,1,1,0,0,1,0,1])
#     y_test=np.array(y_train[i])
#     y_train=list(y_train)
#     y_train.pop(i)
#     y_train=np.array(y_train)
#     print("y train:",y_train)

#     #PET_train,PET_test,PET_vol_train,PET_vol_test=train_test_split(PET_relevant_df, PET_volumes, test_size=0.33, random_state=42)

#     LR.fit(PET_train,PET_vol_train)
#     GNB.fit(PET_train,PET_vol_train)

#     LR_scores = cross_val_score(LR, PET_relevant_df, PET_volumes, scoring="neg_mean_absolute_error", cv=cross_validator)
#     print("LR:", LR_scores.mean(), "+/-", LR_scores.std())

#     GNB_scores = cross_val_score(GNB, PET_relevant_df, PET_volumes, scoring="neg_mean_absolute_error", cv=cross_validator)
#     print("GNB:", GNB_scores.mean(), "+/-", GNB_scores.std())

#     #for LR
#     importance=LR.coef_
#     print("Importance for LR PET-DCE: ",importance)

#     y_pred_LR = LR.fit(PET_train, PET_vol_train).predict(PET_test)
#     y_pred_LRs.append(y_pred_LR)
#     print("LR predicted PET vols: ",y_pred_LR)
#     print("Correct PET vols: ",PET_vol_test)
#     print(f"Number of mislabeled points out of a total %d points : %d" % (PET_test.shape[0], np.sum(PET_vol_test != y_pred_LR)))

#     #for gausian NB
#     y_pred_GNB = GNB.fit(PET_train, PET_vol_train).predict(PET_test)
#     y_pred_GNBs.append(y_pred_GNB)
#     print("GNB predicted PET vols: ",y_pred_GNB)
#     print("Correct PET vols: ",PET_vol_test)
#     print(f"Number of mislabeled points out of a total %d points : %d" % (PET_test.shape[0], np.sum(PET_vol_test != y_pred_GNB)))

#     LR_y_pred_proba = LR.predict_proba(PET_test)
#     GNB_y_pred_proba = GNB.predict_proba(PET_test)

#     LR_y_pred_probas[i:i+length,:]=LR_y_pred_proba
#     GNB_y_pred_probas[i:i+length,:]=GNB_y_pred_proba

#     print("PET vol test:",PET_vol_test,"PET LR pred proba:",LR_y_pred_proba[:,1])
#     LR_fpr, LR_tpr, _ = roc_curve(PET_vol_test, LR_y_pred_proba[:,1])
#     LR_fprs.append(LR_fpr)
#     LR_tprs.append(LR_tpr)
#     print("LR_fpr, LR_tpr: ", LR_fpr, LR_tpr)

#     GNB_fpr, GNB_tpr, _ = roc_curve(PET_vol_test, GNB_y_pred_proba[:,1])
#     GNB_fprs.append(GNB_fpr)
#     GNB_tprs.append(GNB_tpr)
#     print("GNB_fpr, GNB_tpr: ", GNB_fpr, GNB_tpr)

#     """
#     Compute and plot an ROC curve
#     """

#     from sklearn.metrics import roc_curve

#     fpr, tpr, _ = roc_curve(PET_vol_test, LR_y_pred_proba[:,1])
#     fig,ax = plt.subplots(1,1)
#     ax.plot(fpr, tpr)
#     ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
#     plt.title("ROC curve for multimodality Logistic Regression")
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     print("AUC for LR multimodality:",metrics.auc(fpr, tpr))
#     LR_AUCs.append(metrics.auc(LR_fpr, LR_tpr))
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_multimodality_PET_LR.png')

#     fpr, tpr, _ = roc_curve(PET_vol_test, GNB_y_pred_proba[:,1])
#     fig,ax = plt.subplots(1,1)
#     ax.plot(fpr, tpr)
#     ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
#     plt.title("ROC curve for multimodality Gaussian Naive Bayes")
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     print("AUC for GNB multimodality:",metrics.auc(fpr, tpr))
#     GNB_AUCs.append(metrics.auc(GNB_fpr, GNB_tpr))
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_multimodality_PET_GNB.png')
    
#     i=i+1

# """
# Accuracy, sensitivity, specificity
# """
# #Confusion matrix, Accuracy, sensitivity and specificity

# print("y_pred_LRs",y_pred_LRs)
# print("y_pred_GNBs",y_pred_GNBs)

# y_pred_LRs=np.array([0, 0, 1,0, 1, 1,1, 1, 0,0, 1, 1, 1])
# y_pred_GNBs=np.array([0, 0, 1,1, 0, 1,1, 1, 0,0, 0, 0, 0])

# print(f"Number of LR mislabeled points out of a total %d points : %d" % (PET_volumes.shape[0], np.sum(PET_volumes != y_pred_LRs)))
# print(f"Number of GNB mislabeled points out of a total %d points : %d" % (PET_volumes.shape[0], np.sum(PET_volumes != y_pred_GNBs)))

# cm1_LR = confusion_matrix(PET_volumes,y_pred_LRs)
# print('Confusion Matrix LR: \n', cm1_LR)

# total1=sum(sum(cm1_LR))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
# print ('Accuracy LR : ', accuracy1)
# #LRaccuracies.append(accuracy1)

# sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
# print('Sensitivity LR : ', sensitivity1 )
# #LR_sensitivities.append(sensitivity1)

# specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
# print('Specificity LR : ', specificity1)
# #LR_specificities.append(specificity1)

# cm1_GNB = confusion_matrix(PET_volumes,y_pred_GNBs)
# print('Confusion Matrix GNB: \n', cm1_GNB)

# total1=sum(sum(cm1_GNB))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
# print ('Accuracy GNB: ', accuracy1)
# #GNBaccuracies.append(accuracy1)

# sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
# print('Sensitivity GNB: ', sensitivity1 )
# #GNB_sensitivities.append(sensitivity1)

# specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
# print('Specificity GNB: ', specificity1)

# print("average LR AUC:",np.mean(LR_AUCs),"std dev:", np.std(LR_AUCs))
# print("average GNB AUC:",np.mean(GNB_AUCs),"std dev:", np.std(GNB_AUCs))

# """
# Accuracy, sensitivity, specificity
# """

# from sklearn.metrics import confusion_matrix

# #Confusion matrix, Accuracy, sensitivity and specificity

# cm1_LR = confusion_matrix(PET_vol_test,y_pred_LR)
# print('Confusion Matrix LR: \n', cm1_LR)

# total1=sum(sum(cm1_LR))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
# print ('Accuracy LR : ', accuracy1)

# sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
# print('Sensitivity LR : ', sensitivity1 )

# specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
# print('Specificity LR : ', specificity1)

# cm1_GNB = confusion_matrix(PET_vol_test,y_pred_GNB)
# print('Confusion Matrix GNB: \n', cm1_GNB)

# total1=sum(sum(cm1_GNB))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
# print ('Accuracy GNB: ', accuracy1)

# sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
# print('Sensitivity GNB: ', sensitivity1 )

# specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
# print('Specificity GNB: ', specificity1)

# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
# plt.figure()
# plot_confusion_matrix(LR, PET_test, PET_vol_test)
# plt.title("Confusion matrix for multimodality Logistic Regression model")
# #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_combined_LR.png")

# plt.figure()
# plot_confusion_matrix(GNB, PET_test, PET_vol_test)  
# plt.title("Confusion matrix for multimodality Gaussian Naive Bayes model")
# #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_combined_GNB.png")

# """
# Correlation plots
# """

# def CorrPlots(d,method,modality,shortened=False):
#     # Compute the correlation matrix
#     corr = d.corr(method)

#     # Generate a mask for the upper triangle
#     mask = np.triu(np.ones_like(corr, dtype=bool))

#     # Set up the matplotlib figure
#     f, ax = plt.subplots(figsize=(11, 9))

#     # Generate a custom diverging colormap
#     cmap = sns.diverging_palette(230, 20, as_cmap=True)

#     # Draw the heatmap with the mask and correct aspect ratio
#     sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#     fig = plt.gcf(); fig.tight_layout()
#     if shortened==False:
#         plt.savefig(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/{modality}_{method}_heatmap.png")
#     elif shortened==True:
#         plt.savefig(f"/home/alicja/PET-LAB Code/PET-LAB/Radiomics/{modality}_{method}_shortened_heatmap.png")

# #CorrPlots(X_relevant_df,'pearson','multimodality',True)
# #CorrPlots(X_relevant_df,'spearman','multimodality',True)