# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:30:07 2020

@author: Henock
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:11:33 2020

@author: Henock
"""

import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.svm import SVC
from sklearn import metrics as m
import itertools

# set random state
seed=45
np.random.seed(seed)

# manage data
iris= datasets.load_iris()
x=iris.data
y=iris.target
# train test split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=seed)

#===============================================================
#===============================================================
#===============================================================
#===============================================================

#train rf model
rfclf= RandomForestClassifier(random_state=seed)
rfclf.fit(xtrain,ytrain)
rf_ypred=rfclf.predict(xtest)
rf_ypredprob=rfclf.predict_proba(xtest)
rf_confmat=m.confusion_matrix(ytest,rf_ypred)

confsumh = np.sum(rf_confmat, axis=1)
propconfmat = rf_confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i] = 100 * propconfmat[i] / confsumh[i]
rf_ypredconfprob_all=(propconfmat / 100)  
    
print("RF ====> original score, ",m.accuracy_score(ytest,rf_ypred))

#===============================================================
#===============================================================
#===============================================================
#===============================================================

#train svm model
svcclf= SVC(random_state=seed,probability=True)
svcclf.fit(xtrain,ytrain)
svc_ypred=svcclf.predict(xtest)
svc_confmat=m.confusion_matrix(ytest,svc_ypred)
svc_ypredprob=svcclf.predict_proba(xtest)
svc_confmat=m.confusion_matrix(ytest,svc_ypred)

confsumh = np.sum(svc_confmat, axis=1)
propconfmat = svc_confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i] = 100 * propconfmat[i] / confsumh[i]
svc_ypredconfprob_all=(propconfmat / 100)  
    
print("SVM ====> original score, ",m.accuracy_score(ytest,svc_ypred))
 
#===============================================================
#===============================================================
#===============================================================
#===============================================================


predprobalist=[rf_ypredprob,svc_ypredprob]
ypredconfprob_all=[rf_ypredconfprob_all,svc_ypredconfprob_all]


cimc2=np.zeros((30,3))
# # for each classifier
for j in range(2):
#     #for each data point    
    cimc=[]    
    for i in range(30):
        # print(i,ypredconfprob_all[j],predprobalist[j][i])
        
        c1=np.sum(predprobalist[j][i] * ypredconfprob_all[j][0])
        c2=np.sum(predprobalist[j][i] * ypredconfprob_all[j][1])
        c3=np.sum(predprobalist[j][i] * ypredconfprob_all[j][2])
        
        # print([c1,c2,c3])
        cimc.append([c1,c2,c3])
    
    cimc=np.array(cimc)
    ypred2=np.argmax(cimc,axis=1)
    # print((m.accuracy_score(ytest,ypred2)))§
    cimc2 = cimc2 + cimc
    
cimc2=np.array(cimc2)
ypred3=np.argmax(cimc2,axis=1)
print("final cim ==== >",(m.accuracy_score(ytest,ypred3))) 

#===============================================================
#===============================================================
#===============================================================
#===============================================================

eclf1 = VotingClassifier(estimators=[
        ('rfclf', rfclf), ('svcclf', svcclf)], voting='soft')
eclf1 = eclf1.fit(xtrain,ytrain)
eclf1_pred=eclf1.predict(xtest)
print("final softvoting ==== >",m.accuracy_score(ytest,eclf1_pred))

#===============================================================
#===============================================================
#===============================================================
#===============================================================

eclf2 = VotingClassifier(estimators=[
        ('rfclf', rfclf), ('svcclf', svcclf) ], voting='hard')
eclf2 = eclf2.fit(xtrain,ytrain)
eclf2_pred=eclf2.predict(xtest)
print(m.accuracy_score(ytest,eclf2_pred))

#===============================================================
#===============================================================
#===============================================================
#===============================================================

test_list1=svc_ypred
test_list2=rf_ypred


print("=====similarity========")
print( m.accuracy_score(rf_ypred,svc_ypred)) 



temp = pd.DataFrame(data=np.array(ytest), columns=["ytest"])
temp["classifer1_RF"] = rf_ypred
temp["classifer2_svm"] = svc_ypred 
temp["classifer3_CIM"] = ypred3
temp["classifer4_SoftVoting"] = eclf1_pred
temp.to_csv("iris_test.csv", index=False)
temp




















