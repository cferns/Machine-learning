from __future__ import division
import numpy as np
import random
import copy
import csv
import math
#from sklearn.model_selection import KFold
#from sklearn.utils import shuffle
#from sklearn.metrics import f1_score

def write_solutions(algo,predicted_list, fileNAME):
    filepath = './data/data-splits/eval.id'
    file = open(filepath, 'rb')
    data = csv.reader(file)
    example_id_matrix = [row for row in data]
    newList = [['example_id', 'label']]
    if algo == 'perceptron':
        for i in range(len(predicted_list)):
            predicted_list[i] = int((predicted_list[i]+1)/2)
            newList.append([example_id_matrix[i][0],predicted_list[i]])
    elif algo == 'decision tree':
        for i in range(len(predicted_list)):
            predicted_list[i] = int(predicted_list[i])
            newList.append([example_id_matrix[i][0],predicted_list[i]])
    with open(fileNAME, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(newList)

def prediction_Perceptron(matRix, actual_labels, w):
    predicted_list = []
    #rows, cols = matRix.shape
    rows = len(matRix)
    cols = len(matRix[0])
    if (w.size-1)!= cols:
        num_extra_features = cols - (w.size - 1)
        for e in range(num_extra_features):
            w = np.append(w,[0])
    for r_ind in range(rows):
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w, currExample))
        predicted_list.append(y_pred)
        #predicted_list.append(int((y_pred+1)/2))
    return predicted_list

def prediction_Winnow(matRix, actual_labels, w):
    predicted_list = []
    #rows, cols = matRix.shape
    rows = len(matRix)
    cols = len(matRix[0])
    if (w.size)!= cols:
        num_extra_features = cols - (w.size)
        for e in range(num_extra_features):
            w = np.append(w,[0])
    errorCount = 0
    for r_ind in range(rows):
        currExample = matRix[r_ind]
        y_pred = np.sign(np.dot(w, currExample)-cols)
        y_actual = actual_labels[r_ind]
        #Decision
        if y_pred != y_actual:
            errorCount += 1
        predicted_list.append(y_pred)
        #predicted_list.append(int((y_pred+1)/2))
    return predicted_list

def accuracyMETRIC(actual_list, predicted_list):
    errorCount = 0
    numExamples = len(actual_list)
    for indx in range(numExamples):
        if actual_list[indx] != predicted_list[indx]:
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
        #elif actual_list[i] + predicted_list[i] == 0:
        elif actual_list[i] + predicted_list[i] == -2:
            TN += 1
        #elif actual_list[i] == 0 and predicted_list[i] == 1:
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
    return F1

def features_n_labels_matrix(featureMat, labelMat):
    finalmatrix = []
    for i in range(len(featureMat)):
        g = np.append(featureMat[i],labelMat[i])
        finalmatrix.append(g.tolist())
    return finalmatrix

def minusOneToZero(trainLabels):
    finallabels = []
    for i in range(len(trainLabels)):
        finallabels.append((trainLabels[i]+1)/2)
    return finallabels

def split_data(train_matrix, train_actual_labels, num_folds):
    training_lab = copy.deepcopy(train_actual_labels)
    training_lab = training_lab.tolist()
    training_mat = copy.deepcopy(train_matrix)
    training_mat = training_mat.tolist()
    subset_size = int(len(training_mat) / num_folds)

    for i in range(num_folds):
        if i == num_folds - 1:
            temp_test_Matrix = training_mat[(i) * subset_size:]
            temp_train_Matrix = training_mat[0:][:i * subset_size]
            #
            temp_test_Labels = training_lab[(i) * subset_size:]
            temp_train_Labels = training_lab[0:][:i * subset_size]
            continue
        temp_test_Matrix = training_mat[i * subset_size:][:subset_size]
        temp_train_Matrix = training_mat[:i * subset_size] + training_mat[(i + 1) * subset_size:]
        #
        temp_test_Labels = training_lab[i * subset_size:][:subset_size]
        temp_train_Labels = training_lab[:i * subset_size] + training_lab[(i + 1) * subset_size:]

def featureTransformation(trainMatrix, testMatrix, evalMatrix,stdConstant):
    #'below code is for outliers'
    numTrainRows = len(trainMatrix)
    numTrainFeatures = len(trainMatrix[0])
    numTestRows = len(testMatrix)
    numEvalRows = len(evalMatrix)
    for f in range(numTrainFeatures):
        stDev = np.std(trainMatrix[:, f])
        meaN = np.mean(trainMatrix[:, f])
        upLimit = meaN + stDev*stdConstant
        numBuckets = math.ceil(math.sqrt(stDev))
        if numBuckets == 0:
            bucketSize = 1
        else:
            bucketSize = math.ceil(upLimit / numBuckets)
        for r in range(numTrainRows):
            if trainMatrix[r][f] > upLimit:
                trainMatrix[r][f] = upLimit
            trainMatrix[r][f] = math.ceil(trainMatrix[r][f] / bucketSize)
            if r < numTestRows:
                if testMatrix[r][f] > upLimit:
                    testMatrix[r][f] = upLimit
                testMatrix[r][f] = math.ceil(testMatrix[r][f] / bucketSize)
            if r < numEvalRows:
                if evalMatrix[r][f] > upLimit:
                    evalMatrix[r][f] = upLimit
                evalMatrix[r][f] = math.ceil(evalMatrix[r][f] / bucketSize)

            '''
            if trainMatrix[r][f] < lowLimit:
                trainMatrix[r][f] = lowLimit
            if testMatrix[r][f] < lowLimit:
                testMatrix[r][f] = lowLimit
            if evalMatrix[r][f] < lowLimit:
                evalMatrix[r][f] = lowLimit
            '''
    return trainMatrix, testMatrix, evalMatrix



