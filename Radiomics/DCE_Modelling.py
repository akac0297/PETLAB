import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler 

LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
GNB = GaussianNB()

#kf = KFold(n_splits=5)

#DCE_volumes=[0.38054322,   0.12242212,   0.45979818,   0.13357694,  -0.04038771, 0.62797084,   0.87002433,   0.12339629,   0.87001655,   0.70248771,   0.66075925,   0.71046931,  -0.07199637,   0.69350888,   0.95480148, -14.12608326]
DCE_volumes=np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
DCE_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df.csv",index_col=0)
LRaccuracies=[]
GNBaccuracies=[]
LR_AUCs=[]
GNB_AUCs=[]

y_pred_LRs=[]
y_pred_GNBs=[]

LR_y_pred_probas=np.empty((16,2),float)
GNB_y_pred_probas=np.empty((16,2),float)
LR_fprs=[]
LR_tprs=[]
GNB_fprs=[]
GNB_tprs=[]

LR_sensitivities=[]
LR_specificities=[]
GNB_sensitivities=[]
GNB_specificities=[]

test_splits=[[0,1,5],[2,3,6],[4,7,8],[9,10,11,12],[13,14,15]]
#test_splits=[[0,1,5,2,9],[3,10,11,12,4,6],[13,14,15,7,8]]
train_splits=[[i for i in range(16) if i not in j] for j in test_splits]

for i in range(16):
    train_indices = list(range(16))
    train_indices.remove(i)
    df_train = DCE_relevant_df.iloc[train_indices]
    DCE_vol_train=DCE_volumes[train_indices]
    df_test = DCE_relevant_df.iloc[i]
    DCE_vol_test=DCE_volumes[i]

    ss = StandardScaler()
    ss.fit(df_train)
    X_train = ss.transform(df_train)
    X_test = ss.transform([df_test])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
    y_test=np.array(y_train[i])
    y_train=list(y_train)
    y_train.pop(i)
    y_train=np.array(y_train)
    #print("y train:",y_train)
    LR.fit(X_train, y_train)
    LR.predict_proba(X_test)

    LR.fit(X_train,DCE_vol_train)
    GNB.fit(X_train,DCE_vol_train)

    y_pred_LR = LR.predict(X_test)

    #for gausian NB
    y_pred_GNB = GNB.predict(X_test)

    LR_y_pred_proba = LR.predict_proba(X_test)
    GNB_y_pred_proba = GNB.predict_proba(X_test)

    y_pred_LRs.append(y_pred_LR)
    y_pred_GNBs.append(y_pred_GNB)

    LR_y_pred_probas[i,:]=LR_y_pred_proba
    GNB_y_pred_probas[i,:]=GNB_y_pred_proba

print(f"Number of LR mislabeled points out of a total {DCE_volumes.shape[0]} points : {np.sum(DCE_volumes != np.array(y_pred_LRs).reshape(1,-1)[0])}")
print(f"Number of GNB mislabeled points out of a total {DCE_volumes.shape[0]} points : {np.sum(DCE_volumes != np.array(y_pred_GNBs).reshape(1,-1)[0])}")

"""
Compute and plot an ROC curve
"""

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(DCE_volumes, LR_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Logistic Regression in DCE-MRI")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for LR in DCE-MRI:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_LR.png')

fpr, tpr, _ = roc_curve(DCE_volumes, GNB_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Gaussian Naive Bayes in DCE-MRI")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for GNB in DCE-MRI:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_GNB.png')

"""
Accuracy, sensitivity, specificity
"""

from sklearn.metrics import confusion_matrix

#Confusion matrix, Accuracy, sensitivity and specificity

cm1_LR = confusion_matrix(DCE_volumes,y_pred_LRs)
print('Confusion Matrix LR: \n', cm1_LR)

total1=sum(sum(cm1_LR))
##from confusion matrix calculate accuracy
accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
print ('Accuracy LR : ', accuracy1)

sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
print('Sensitivity LR : ', sensitivity1 )

specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
print('Specificity LR : ', specificity1)

cm1_GNB = confusion_matrix(DCE_volumes,y_pred_GNBs)
print('Confusion Matrix GNB: \n', cm1_GNB)

total1=sum(sum(cm1_GNB))
##from confusion matrix calculate accuracy
accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
print ('Accuracy GNB: ', accuracy1)

sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
print('Sensitivity GNB: ', sensitivity1 )

specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
print('Specificity GNB: ', specificity1)

# for train_index, test_index in zip(train_splits,test_splits):
#     print(test_index)
#     length=len(test_splits[i])
#     print("length:",length)
#     train_index=[int(i) for i in train_index]
#     test_index=[int(i) for i in test_index]
#     DCE_train, DCE_test = DCE_relevant_df.iloc[train_index].values, DCE_relevant_df.iloc[test_index].values
#     DCE_vol_train, DCE_vol_test = DCE_volumes[train_index], DCE_volumes[test_index]
#     #X_train, X_test, y_train, y_test = train_test_split(X_relevant, y, test_size=0.33, random_state=42)
#     #PET_train,PET_test,PET_vol_train,PET_vol_test=train_test_split(PET_relevant_df, PET_volumes, test_size=0.33, random_state=42)
#     #DCE_train,DCE_test,DCE_vol_train,DCE_vol_test=train_test_split(DCE_relevant_df, DCE_volumes, test_size=0.33, random_state=42)

#     ss = StandardScaler()
#     ss.fit(DCE_train)
#     X_train = ss.transform(DCE_train)
#     X_test = ss.transform(DCE_test)
#     y_train = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
#     y_test=np.array(y_train[i])
#     y_train=list(y_train)
#     y_train.pop(i)
#     y_train=np.array(y_train)
#     print("y train:",y_train)

#     #model.fit(X_train, y_train)
#     print("train dataframe:",DCE_train)
#     LR.fit(DCE_train,DCE_vol_train)
#     GNB.fit(DCE_train,DCE_vol_train)

#     LR_scores = cross_val_score(LR, DCE_relevant_df, DCE_volumes, scoring="neg_mean_absolute_error")
#     print("LR:", LR_scores.mean(), "+/-", LR_scores.std())

#     GNB_scores = cross_val_score(GNB, DCE_relevant_df, DCE_volumes, scoring="neg_mean_absolute_error")
#     print("GNB:", GNB_scores.mean(), "+/-", GNB_scores.std())

#     #for LR
#     importance=LR.coef_
#     print("Importance for LR DCE: ",importance)

#     y_pred_LR = LR.predict(DCE_test)
#     y_pred_LRs.append(y_pred_LR)
#     print("LR predicted DCE vols: ",y_pred_LR)
#     print("Correct DCE vols: ",DCE_vol_test)
#     print(f"Number of mislabeled points out of a total %d points : %d" % (DCE_test.shape[0], np.sum(DCE_vol_test != y_pred_LR)))

#     #for gausian NB
#     y_pred_GNB = GNB.predict(DCE_test)
#     y_pred_GNBs.append(y_pred_GNB)
#     print("GNB predicted DCE vols: ",y_pred_GNB)
#     print("Correct DCE vols: ",DCE_vol_test)
#     print(f"Number of mislabeled points out of a total %d points : %d" % (DCE_test.shape[0], np.sum(DCE_vol_test != y_pred_GNB)))

#     #for i,v in enumerate(importance):
#     #    print('Feature: %0d, Score: %.5f' % (i,v))

#     #LR_y_pred = LR.predict(PET_test)
#     #print(LR_y_pred)

#     LR_y_pred_proba = LR.predict_proba(DCE_test)
#     print("LR Pred proba for DCE:",LR_y_pred_proba)
#     GNB_y_pred_proba = GNB.predict_proba(DCE_test)
#     print("GNB Pred proba for DCE:",GNB_y_pred_proba)

#     LR_y_pred_probas[i:i+length,:]=LR_y_pred_proba
#     GNB_y_pred_probas[i:i+length,:]=GNB_y_pred_proba

#     print("DCE vol test:",DCE_vol_test,"DCE LR pred proba:",LR_y_pred_proba[:,1])
#     LR_fpr, LR_tpr, _ = roc_curve(DCE_vol_test, LR_y_pred_proba[:,1])
#     LR_fprs.append(LR_fpr)
#     LR_tprs.append(LR_tpr)
#     print("LR_fpr, LR_tpr: ", LR_fpr, LR_tpr)

#     GNB_fpr, GNB_tpr, _ = roc_curve(DCE_vol_test, GNB_y_pred_proba[:,1])
#     GNB_fprs.append(GNB_fpr)
#     GNB_tprs.append(GNB_tpr)
#     print("GNB_fpr, GNB_tpr: ", GNB_fpr, GNB_tpr)

#     """
#     Compute and plot an ROC curve
#     """

#     fig,ax = plt.subplots(1,1)
#     ax.plot(LR_fpr, LR_tpr)
#     ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
#     plt.title("ROC curve for Logistic Regression in DCE-MRI")
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     print("AUC for LR in DCE:",metrics.auc(LR_fpr, LR_tpr))
#     LR_AUCs.append(metrics.auc(LR_fpr, LR_tpr))
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_LR.png')

#     #print("DCE volumes",DCE_volumes)
#     fig,ax = plt.subplots(1,1)
#     ax.plot(GNB_fpr, GNB_tpr)
#     ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
#     plt.title("ROC curve for Gaussian Naive Bayes in DCE-MRI")
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     print("AUC for GNB in DCE:",metrics.auc(GNB_fpr, GNB_tpr))
#     GNB_AUCs.append(metrics.auc(GNB_fpr, GNB_tpr))
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_GNB.png')

#     i+=1

# print("LR AUCs:",LR_AUCs)
# print("GNB AUCs:",GNB_AUCs)

# """
# Accuracy, sensitivity, specificity
# """
# #Confusion matrix, Accuracy, sensitivity and specificity

# print("y_pred_LRs",y_pred_LRs)
# print("y_pred_GNBs",y_pred_GNBs)

# y_pred_LRs=np.array([0, 1, 1,1, 0, 1,0, 1, 0,1, 1, 1, 0,1, 0, 0])
# y_pred_GNBs=np.array([0, 1, 1,0, 0, 1,0, 0, 0,1, 1, 1, 0,1, 0, 0])

# print(f"Number of LR mislabeled points out of a total %d points : %d" % (DCE_volumes.shape[0], np.sum(DCE_volumes != y_pred_LRs)))
# print(f"Number of GNB mislabeled points out of a total %d points : %d" % (DCE_volumes.shape[0], np.sum(DCE_volumes != y_pred_GNBs)))

# cm1_LR = confusion_matrix(DCE_volumes,y_pred_LRs)
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

# cm1_GNB = confusion_matrix(DCE_volumes,y_pred_GNBs)
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
# #GNB_specificities.append(specificity1)

# print("average LR AUC:",np.mean(LR_AUCs),"std dev:", np.std(LR_AUCs))
# print("average GNB AUC:",np.mean(GNB_AUCs),"std dev:", np.std(GNB_AUCs))

# #mean_LR_accuracy=np.mean(LRaccuracies)
# #mean_GNB_accuracy=np.mean(GNBaccuracies)

# #print("mean LR accuracy for DCE-MRI:",mean_LR_accuracy, "std dev:", np.std(LRaccuracies))
# #print("mean GNB accuracy for DCE-MRI:",mean_GNB_accuracy,"std dev:",np.std(GNBaccuracies))

# #print("mean sensitivity for LR:",np.mean(LR_sensitivities),"mean specificity for LR:",np.mean(LR_specificities))
# #print("mean sensitivity for GNB:",np.mean(GNB_sensitivities),"mean specificity for GNB:",np.mean(GNB_specificities))

# #print("LR tpr:", LR_tprs,"LR fpr:",LR_fprs)
# #print("GNB tpr:", GNB_tprs,"GNB fpr:",GNB_fprs)

# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
# plt.figure()
# plot_confusion_matrix(LR, DCE_test, DCE_vol_test)
# plt.title("Confusion matrix for DCE-MRI Logistic Regression model")
# #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_DCE-MRI_LR.png")

# plt.figure()
# plot_confusion_matrix(GNB, DCE_test, DCE_vol_test)  
# plt.title("Confusion matrix for DCE-MRI Gaussian Naive Bayes model")
# #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_DCE-MRI_GNB.png")
