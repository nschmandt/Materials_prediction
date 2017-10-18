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

model = Sequential()
s = 'relu'
model.add(Dense(100, activation=s, input_dim=158))
model.add(Dropout(.5))
model.add(Dense(100, activation=s))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
keras.optimizers.RMSprop(lr=.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batches = 20

# Loop over all 9 prediction vectors
for j in range(9):

    f_score_values = []
    true_pos_scores = []
    false_pos_scores = []
    false_neg_scores = []

    # Loop over the five different cross-validation sets
    for i in range(5):
        temp = list(range(5))
        temp.remove(i)
        # this is our set that is removed from training for validation
        val = training_nonoble[i::5]
        # training set consists of the other values
        train_X = [training_nonoble[temp[0]::5], training_nonoble[temp[1]::5], training_nonoble[temp[2]::5],
                   training_nonoble[temp[3]::5]]
        train_X = pd.concat(train_X)

        # Fit the model
        temp = np.array(train_X[col_names])
        model.fit(temp, train_X['stabilityVec%s' % j], batch_size=batches, epochs=2000, verbose=0)
        # Predict the model
        temp = np.array(val[col_names])
        nn_prediction = model.predict(temp)
        # Turn it back into a list
        nn_prediction = np.ndarray.tolist(nn_prediction)
        nn_prediction = [int(nn_prediction[i][0]) for i in range(len(nn_prediction))]

        # Check for false positives, false negatives and true positives
        temp = nn_prediction - val['stabilityVec%s' % j]
        false_pos = np.sum(temp == 1)
        false_neg = np.sum(temp == -1)
        temp = nn_prediction + val['stabilityVec%s' % j]
        true_pos = np.sum(temp == 2)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_score = 2 * precision * recall / (precision + recall)
        f_score_values.append(f_score)
        true_pos_scores.append(true_pos)
        false_pos_scores.append(false_pos)
        false_neg_scores.append(false_neg)
    print('for vector %s, the mean f-score is %.2f' % (str(j), np.mean(f_score_values)))
    print('True Positives: %d, False Positives: %d, False Negatives: %d' % (np.mean(true_pos_scores), \
                                                                            np.mean(false_pos_scores),
                                                                            np.mean(false_neg_scores)))