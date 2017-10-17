import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#For a Logistic Regression, let's start by examining how well it fits the cross-validated training sets.
#looping over all 9 of the classification predictions
for j in range(9):
    f_score_values=[]
    true_pos_scores=[]
    false_pos_scores=[]
    false_neg_scores=[]
    #Cross Validation Groups
    for i in range(5):
        temp=list(range(5))
        temp.remove(i)
        #this is our set that is removed from training for validation
        val=training_nonoble[i::5]
        #training set consists of the other values
        train_X=[training_nonoble[temp[0]::5], training_nonoble[temp[1]::5], training_nonoble[temp[2]::5], training_nonoble[temp[3]::5]]
        train_X=pd.concat(train_X)
        #declare the model
        model=LogisticRegression()
        #fit the model
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        #make a prediction
        lr_prediction=fit.predict(train_X[col_names])
        temp=lr_prediction-train_X['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=lr_prediction+train_X['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    print('for vector position %s, the f_scores for cross-validation are as follows:' % str(j+1))
    print(f_score_values)
    print('True Positives:')
    print(true_pos_scores)
    print('False Positives')
    print(false_pos_scores)
    print('False Neg (missed positives)')
    print(false_neg_scores)


#It's fitting better than I thought it would. Let's see how it does with the test sets.
for j in range(9):
    f_score_values=[]
    true_pos_scores=[]
    false_pos_scores=[]
    false_neg_scores=[]
    #Cross Validation Groups
    for i in range(5):
        temp=list(range(5))
        temp.remove(i)
        #this is our set that is removed from training for validation
        val=training_nonoble[i::5]
        #training set consists of the other values
        train_X=[training_nonoble[temp[0]::5], training_nonoble[temp[1]::5], training_nonoble[temp[2]::5], training_nonoble[temp[3]::5]]
        train_X=pd.concat(train_X)
        #declare the model
        model=LogisticRegression()
        #fit the model
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        #make a prediction
        lr_prediction=fit.predict(val[col_names])
        temp=lr_prediction-val['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=lr_prediction+val['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    print('for vector position %s, the f_scores for cross-validation are as follows:' % str(j+1))
    print(f_score_values)
    print('True Positives:')
    print(true_pos_scores)
    print('False Positives')
    print(false_pos_scores)
    print('False Neg (missed positives)')
    print(false_neg_scores)