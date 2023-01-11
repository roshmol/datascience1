# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 19:19:22 2022

@author: ASUS
"""

import pandas as pd
df=pd.read_csv("Social_Network_Ads.csv")
x=df.iloc[:,0:2].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
models={'logistic':LogisticRegression(),'decision_tree':DecisionTreeClassifier(),'random_forest':RandomForestClassifier()}
res=[]
for i in  models:
      model=models[i]
      scores=cross_val_score(model,x,y,cv=5)
      a=(i,scores.mean())
      res.append(a)

models={'Gaussian':GaussianNB(),'bernouli':BernoulliNB(),'multi':MultinomialNB()}
res=[]
for i in models:
    model1=models[i]
    scores=cross_val_score(model1,x,y,cv=5)
    b=(i,scores.mean())
    res.append(b)


rf_params={'n_estimators':[10,20,30,40]}
#svm_params={'gamma':[1,2,5,10,100],'C':[1,5,10,100],'kernel':['poly','linear','rbf','sigmoid']}
Ds_params={'criterion':['gini','entropy']} 

model_details={'rf':{'model':RandomForestClassifier(),'params':rf_params},'Ds':{'model':DecisionTreeClassifier(),'params':Ds_params}}

#'svm':{'model':SVC(),'params':svm_params}
def get_best(x,y):
    result=[]
    for model_name in model_details:
        model_dict=model_details.get(model_name)
        model=model_dict.get('model')
        params=model_dict.get('params')
        cv=GridSearchCV(model,params,cv=5,return_train_score=True)
        cv.fit(x,y)
        bp=cv.best_params_
        bs=cv.best_score_
        t_res={'model':model_name,'best_params':bp,'best_score':bs}
        result.append(t_res)
    res_df=pd.DataFrame(result)
    best_model=res_df[res_df['best_score']==res_df['best_score'].max()]
    return res_df,best_model
    
get_best(x,y)
res_df,best_model=get_best(x,y)
res_df
best_model
