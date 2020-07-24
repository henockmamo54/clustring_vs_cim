# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:30:00 2020

@author: Henock
"""

import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics as m
import itertools

# set random state
seed=45
np.random.seed(seed)

iris= datasets.load_iris()
x=iris.data
y=iris.target

# train test split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=seed)

#train model
rfclf= RandomForestClassifier(random_state=seed)
rfclf.fit(xtrain,ytrain)
ypred=rfclf.predict(xtest)
print("====> original score, ",m.accuracy_score(ytest,ypred))

comb1=[0,1,2,3]
comb2=list(itertools.combinations(comb1,2))
comb3=list(itertools.combinations(comb1,3))
comb4=list(itertools.combinations(comb1,4))
comb=comb1+comb2+comb3+comb4

acclist=[]
predprobalist=[]
confmatrxlist=[]
ypredconfprob_all=[]



for f in comb:
    rfclf= RandomForestClassifier(random_state=seed)
    if(type(f)==int):
        rfclf.fit(xtrain[:,f].reshape(-1, 1),ytrain)
        ypred=rfclf.predict(xtest[:,f].reshape(-1, 1))
        ypredprob=rfclf.predict_proba(xtest[:,f].reshape(-1, 1))
    else:
        rfclf.fit(xtrain[:,f],ytrain)
        ypred=rfclf.predict(xtest[:,f])
        ypredprob=rfclf.predict_proba(xtest[:,f])
        
    accuracy=(m.accuracy_score(ytest,ypred))
    acclist.append(accuracy)
    predprobalist.append(ypredprob)
    
    confmat=m.confusion_matrix(ytest,ypred)
    confmatrxlist.append(confmat)
    
    confsumh = np.sum(confmat, axis=1)
    propconfmat = confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[i] = 100 * propconfmat[i] / confsumh[i]
    ypredconfprob_all.append(propconfmat / 100)
    


cimc2=np.zeros((30,3))
# # for each classifier
for j in range(15):
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
    print((m.accuracy_score(ytest,ypred2)))
    cimc2 = cimc2 + cimc
    
print("final cim ==== >")
cimc2=np.array(cimc2)
ypred3=np.argmax(cimc2,axis=1)
print((m.accuracy_score(ytest,ypred3))) 









