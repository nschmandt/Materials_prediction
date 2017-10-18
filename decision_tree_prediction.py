import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import keras

#Let's start with a Random Forest Predictor, with 5 cross-validation sets
#using the 2nd prediction set, let's see how many nodes are necessary for a good fit to the training set.

train_f_score=[]
j=2
for k in [5, 10, 20, 40, 60, 80, 100, 150]:
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
        model=RandomForestClassifier(k, n_jobs=-1)
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        rf_prediction=fit.predict(train_X[col_names])
        temp=rf_prediction-train_X['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=rf_prediction+train_X['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    train_f_score.append(np.mean(f_score_values))
    print('with %s nodes, the mean f-score is %.2f' % (str(k), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
            np.mean(false_pos_scores), np.mean(false_neg_scores)))

# Wow, nearly perfect fits for even the smallest set of nodes. Let's see how they perform on the test set.

test_f_score = []
j = 2
for k in [5, 10, 20, 40, 60, 80, 100, 150]:
    f_score_values = []
    true_pos_scores = []
    false_pos_scores = []
    false_neg_scores = []
    # Cross Validation Groups
    for i in range(5):
        temp = list(range(5))
        temp.remove(i)
        # this is our set that is removed from training for validation
        val = training_nonoble[i::5]
        # training set consists of the other values
        train_X = [training_nonoble[temp[0]::5], training_nonoble[temp[1]::5], training_nonoble[temp[2]::5],
                   training_nonoble[temp[3]::5]]
        train_X = pd.concat(train_X)
        model = RandomForestClassifier(k, n_jobs=-1)
        fit = model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        rf_prediction = fit.predict(val[col_names])
        temp = rf_prediction - val['stabilityVec%s' % j]
        false_pos = sum(temp == 1)
        false_neg = sum(temp == -1)
        temp = rf_prediction + val['stabilityVec%s' % j]
        true_pos = sum(temp == 2)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_score = 2 * precision * recall / (precision + recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    test_f_score.append(np.mean(f_score_values))
    print('with %s nodes, the mean f-score is %.2f' % (str(k), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
                                                                            np.mean(false_pos_scores),
                                                                            np.mean(false_neg_scores)))


#the random forest seems to be doing very well with this data set, and there doesn't seem to be any evidence of
#overfitting.
plt.plot([5, 10, 20, 40, 60, 80, 100, 150], train_f_score, label='train')
plt.plot([5, 10, 20, 40, 60, 80, 100, 150], test_f_score, label='test')
plt.legend()
plt.show()


#I'm going to try again with a more sparsely populated vector here:

train_f_score=[]
j=1
for k in [5, 10, 20, 40, 60, 80, 100, 150]:
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
        model=RandomForestClassifier(k, n_jobs=-1)
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        rf_prediction=fit.predict(train_X[col_names])
        temp=rf_prediction-train_X['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=rf_prediction+train_X['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    train_f_score.append(np.mean(f_score_values))
    print('with %s nodes, the mean f-score is %.2f' % (str(k), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
            np.mean(false_pos_scores), np.mean(false_neg_scores)))


#Wow, nearly perfect fits for even the smallest set of nodes. Let's see how they perform on the test set.

test_f_score=[]
j=1
for k in [5, 10, 20, 40, 60, 80, 100, 150]:
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
        model=RandomForestClassifier(k, n_jobs=-1)
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        rf_prediction=fit.predict(val[col_names])
        temp=rf_prediction-val['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=rf_prediction+val['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    test_f_score.append(np.mean(f_score_values))
    print('with %s nodes, the mean f-score is %.2f' % (str(k), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
            np.mean(false_pos_scores), np.mean(false_neg_scores)))


#Again, this is a pretty good performance, with no evidence of overfitting.
plt.plot([5, 10, 20, 40, 60, 80, 100, 150], train_f_score, label='train')
plt.plot([5, 10, 20, 40, 60, 80, 100, 150], test_f_score, label='test')
plt.legend()
plt.show()

# Let's go for 100 nodes and loop over all 9 of the vectors.
for j in range(9):
    f_score_values = []
    true_pos_scores = []
    false_pos_scores = []
    false_neg_scores = []
    # Cross Validation Groups
    for i in range(5):
        temp = list(range(5))
        temp.remove(i)
        # this is our set that is removed from training for validation
        val = training_nonoble[i::5]
        # training set consists of the other values
        train_X = [training_nonoble[temp[0]::5], training_nonoble[temp[1]::5], training_nonoble[temp[2]::5],
                   training_nonoble[temp[3]::5]]
        train_X = pd.concat(train_X)
        model = RandomForestClassifier(100, n_jobs=-1)
        fit = model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        rf_prediction = fit.predict(val[col_names])
        temp = rf_prediction - val['stabilityVec%s' % j]
        false_pos = sum(temp == 1)
        false_neg = sum(temp == -1)
        temp = rf_prediction + val['stabilityVec%s' % j]
        true_pos = sum(temp == 2)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_score = 2 * precision * recall / (precision + recall)
        if np.isnan(f_score): f_score = 0
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    print('for vector %s, the mean f-score is %.2f' % (str(j), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
                                                                            np.mean(false_pos_scores),
                                                                            np.mean(false_neg_scores)))


#Not bad! Now, let's give gradient boosting a try.
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
        model=GradientBoostingClassifier()
        fit=model.fit(train_X[col_names], train_X['stabilityVec%s' % j])
        gb_prediction=fit.predict(val[col_names])
        temp=gb_prediction-val['stabilityVec%s' % j]
        false_pos=sum(temp==1)
        false_neg=sum(temp==-1)
        temp=gb_prediction+val['stabilityVec%s' % j]
        true_pos=sum(temp==2)
        precision=true_pos/(true_pos+false_pos)
        recall=true_pos/(true_pos+false_neg)
        f_score=2*precision*recall/(precision+recall)
        if np.isnan(f_score): f_score=0
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    print('for vector %s, the mean f-score is %.2f' % (str(j), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
            np.mean(false_pos_scores), np.mean(false_neg_scores)))



#The two are pretty similar, which makes sense. I'll go ahead and make a prediction on the test dataset, though
#I am still waiting on some results for neural nets before I decide which to ultimately use for my submission.

#Test data was processed identically to the training data and normalized the same way, removing noble gases and
#then dividing by the max value of each column.

#Sanity check ratios are included.

for j in range(9):
    model=GradientBoostingClassifier()
    fit=model.fit(training_nonoble[col_names], training_nonoble['stabilityVec%s' % j])
    gb_prediction=fit.predict(training_nonoble[col_names])
    print('Training set vector %s prediction ratio is %.2f' %(j, np.sum(gb_prediction)/len(gb_prediction)))
    gb_prediction=fit.predict(test[col_names])
    print('Test set vector %s prediction ratio is %.2f' %(j, np.sum(gb_prediction)/len(gb_prediction)))
    test['stabilityVec%s' % j] = gb_prediction

# Those numbers look reasonable.
# Setting noble gas predictions to zero
for i in range(len(test)):
    if test['formulaA'][i] in ['Ne', 'Ar', 'Kr', 'He', 'Xe']:
        for j in range(9):
            test['stabilityVec%s' % j][i] = 0

    if test['formulaB'][i] in ['Ne', 'Ar', 'Kr', 'He', 'Xe']:
        for j in range(9):
            test['stabilityVec%s' % j][i] = 0


#reformat the data into a .1 float string of values
test['stabilityVec'][i]='[1.0,'
for i in range(len(test)):
    for j in range(9):
        test['stabilityVec'][i]+='%.1f,' % test['stabilityVec%s' % j][i]
    test['stabilityVec'][i]+='1.0]'


#Hope its reasonably right! Time to save the values
test.to_csv('test_data_output.csv')