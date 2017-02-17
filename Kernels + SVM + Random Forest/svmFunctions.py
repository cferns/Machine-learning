from __future__ import division
import numpy as np
import random
import copy
import csv
import math
from random import shuffle

def read_file(filepath):
    file = open(filepath, 'rb')
    data = csv.reader(file, delimiter=' ')
    tempMatrix = [row for row in data]
    matrix = np.array(tempMatrix)
    num_examples = len(matrix)
    numCols = len(matrix[0])
    if numCols == 501:
        numCols = 500
    finalMatrix = np.zeros((num_examples,numCols), dtype=np.float)
    for i in range(num_examples):
        for j in range(numCols):
            finalMatrix[i][j] = float(matrix[i][j])
    finalMatrix.tolist()
    return finalMatrix

def SVM(OrigMatrix, labels, epochs,C, gamma_0):
    labels = np.array(labels)
    matrix = copy.deepcopy(OrigMatrix)
    numCols = len(matrix[0])
    numExamples = len(matrix)
    w = np.zeros(numCols+1, dtype=np.float)
    for ep in range(epochs):
        for r_ind in range(numExamples):
            gamma = gamma_0 / (1 + gamma_0 * (r_ind+1) / C)
            currExample = np.append([1], matrix[r_ind])
            y_actual = labels[r_ind]
            product = y_actual*np.dot(w,currExample)
            if product <= 1:
                w = (1-gamma)*w + gamma*C*y_actual*currExample
            else:
                w = (1-gamma)*w
        matrix, labels = shuffleFunction(matrix, labels)
    return w

def shuffleFunction(a, b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size // len(a)].reshape(a.shape)
    b2 = c[:, a.size // len(a):].reshape(b.shape)
    return a2, b2

def prediction(matRix, actual_labels, w):
    predicted_list = []
    rows = len(matRix)
    cols = len(matRix[0])
    if (w.size-1) != cols:
        num_extra_features = cols - (w.size - 1)
        for e in range(num_extra_features):
            w = np.append(w,[0])
    for r_ind in range(rows):
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w, currExample))
        predicted_list.append(y_pred)
    return predicted_list

def accuracyMETRIC(actual_list, predicted_list):
    errorCount = 0
    numExamples = len(actual_list)
    for indx in range(numExamples):
        if actual_list[indx]*predicted_list[indx] <= 0:
            errorCount += 1
    accuracy = round(float(((numExamples - errorCount) / numExamples) * 100), 2)
    return accuracy

def f1METRIC(actual_list, predicted_list):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(actual_list)):
        if actual_list[i] + predicted_list[i] == 2:
            TP += 1
        elif actual_list[i] + predicted_list[i] == -2:
            TN += 1
        elif actual_list[i] == -1 and predicted_list[i] == 1:
            FP += 1
        else:
            FN += 1
    if (TP+FP)==0:
        precision = 1
    else:
        precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall / (precision + recall)
    return precision, recall, F1

def crossValidation(OrigMatrix,OrigLabels,Clist,Rlist,num_folds,epochs):
    store = []
    av_accuracy_list = []
    for gamma in Rlist:
        for C in Clist:
            matrix = copy.deepcopy(OrigMatrix)
            labels = copy.deepcopy(OrigLabels)
            accuracy_list = []
            #
            subset_size = int(len(matrix) / num_folds)
            for i in range(num_folds):
                if i == 0:
                    test_Matrix = matrix[i * subset_size: ][ :subset_size]
                    train_Matrix =  matrix[(i + 1) * subset_size:]
                    test_Labels = labels[i * subset_size:][:subset_size]
                    train_Labels = labels[(i + 1) * subset_size:]
                elif i == num_folds - 1:
                    test_Matrix = matrix[(i) * subset_size:]
                    train_Matrix = matrix[0:][:i * subset_size]
                    test_Labels = labels[(i) * subset_size:]
                    train_Labels = labels[0:][:i * subset_size]
                else:
                    test_Matrix = matrix[i * subset_size:][:subset_size]
                    train_Matrix = np.concatenate((matrix[:i * subset_size],matrix[(i + 1) * subset_size:]), axis=0)
                    test_Labels = labels[i * subset_size:][:subset_size]
                    train_Labels = np.concatenate((labels[:i * subset_size],labels[(i + 1) * subset_size:]), axis=0)
                # Training
                w = SVM(train_Matrix, train_Labels, epochs, C, gamma)
                predicted_labels = prediction(test_Matrix, test_Labels, w)
                accuracy = accuracyMETRIC(test_Labels, predicted_labels)
                accuracy_list.append(accuracy)
            av_accuracy = np.mean(accuracy_list)
            av_accuracy_list.append(av_accuracy)
            store.append([gamma, C, av_accuracy])
    bestAccuray = max(av_accuracy_list)
    max_index = av_accuracy_list.index(bestAccuray)
    bestGamma = store[max_index][0]
    bestC = store[max_index][1]
    #
    print '\tgamma    C        Av accuracy %'
    for st in range(len(store)):
        print '\t',store[st][0],'  ',store[st][1], '       ',store[st][2]
    return store,bestGamma,bestC,bestAccuray

