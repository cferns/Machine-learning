from __future__ import division
import numpy as np
import random
import copy
import csv
import math

def readFile(filepath):
    file = open(filepath, 'rb')
    data = csv.reader(file, delimiter=' ')
    matrix = [row for row in data]
    for mDat in matrix:
        if mDat[-1] == '':
            del mDat[-1]
    labels = []
    Totalmatrix = []
    maxFtrCol = 0
    #startWzero = 'no'
    for i in range(len(matrix)):
        lineData = csv.reader(matrix[i], delimiter=':')
        rowmatrix = [row for row in lineData]
        T2 = [map(int, x) for x in rowmatrix]
        labels.append(T2[0][0])
        #if T2[1][0] == 0:
            #startWzero = 'yes'
        for j in range(1, len(T2)):
            tempo = T2[j][0]
            if ((tempo) > maxFtrCol):
                maxFtrCol = tempo
        Totalmatrix.append(T2)

    #if startWzero is 'yes':
    if Totalmatrix[0][1][0] == 0:
        finalMatrix = np.zeros((len(matrix), maxFtrCol + 1))
        for i in range(len(Totalmatrix)):
            for j in range(1, (len(Totalmatrix[i]))):
                finalMatrix[i][Totalmatrix[i][j][0]] = Totalmatrix[i][j][1]
    else:
        finalMatrix = np.zeros((len(matrix), maxFtrCol))
        for i in range(len(Totalmatrix)):
            for j in range(1, (len(Totalmatrix[i]))):
                finalMatrix[i][Totalmatrix[i][j][0] - 1] = 1
    labels = np.asarray(labels)
    return finalMatrix, labels

def logisticRegression(OrigMatrix, labels, epochs, sigma, r):
    costFuncList = []
    labels = np.array(labels)
    matrix = copy.deepcopy(OrigMatrix)
    numCols = len(matrix[0])
    numExamples = len(matrix)
    w = np.zeros(numCols+1, dtype=np.float)
    for ep in range(epochs):
        costFunc = 0
        for r_ind in range(numExamples):
            xi = np.append([1], matrix[r_ind])
            yi = labels[r_ind]
            eComponent = math.exp(yi * np.dot(w, xi))
            w = w + r*(yi*xi/( eComponent + 1) -2*w/(sigma**2))
            #costFunc = costFunc + math.log( 1 + math.exp(-yi*np.dot(w, xi))) + np.dot(w, w)/sigma**2
        matrix, labels = shuffleFunction(matrix, labels)

        costFunc = 0
        for r_ind in range(numExamples):
            xi = np.append([1], matrix[r_ind])
            yi = labels[r_ind]
            costFunc = costFunc + math.log(1 + math.exp(-yi*np.dot(w, xi)))
        costFunc = costFunc + np.dot(w, w)/(sigma**2)

        costFuncList.append(costFunc)
    return w, costFuncList

def shuffleFunction(a, b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size // len(a)].reshape(a.shape)
    b2 = c[:, a.size // len(a):].reshape(b.shape)
    return a2, b2

def prediction(matRix, w):
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

def crossValidation(OrigMatrix, OrigLabels, SigmaList, num_folds, epochs, rate):
    store = []
    av_accuracy_list = []
    for sigma in SigmaList:
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
            w, dummy = logisticRegression(train_Matrix, train_Labels, epochs, sigma, rate)
            predicted_labels = prediction(test_Matrix, w)
            accuracy = accuracyMETRIC(test_Labels, predicted_labels)
            accuracy_list.append(accuracy)
        av_accuracy = np.mean(accuracy_list)
        av_accuracy_list.append(av_accuracy)
        store.append([sigma, av_accuracy])
    bestAccuray = max(av_accuracy_list)
    max_index = av_accuracy_list.index(bestAccuray)
    bestSigma = store[max_index][0]
    #
    print '\tsigma    Average accuracy(%)'
    for st in range(len(store)):
        print '\t ',store[st][0],'    ',store[st][1]
    return store, bestSigma