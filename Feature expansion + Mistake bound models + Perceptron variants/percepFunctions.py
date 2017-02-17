from __future__ import division
import numpy as np
import random
import copy
import csv
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
#scikit learn libraries are not used even though the code lines are typed here


def readFile(filepath):
    csrMatrix, labelsTemp = load_svmlight_file(filepath)
    #following line is to convert folat64 to int32; not required
    labels = np.array(labelsTemp)
    matRix = csrMatrix.toarray()
    #rowsM, colsM = matRix.shape
    return matRix, labels
'''

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
                # change to finalMatrix[i][Totalmatrix[i][j][0] - 1] = Totalmatrix[i][j][1]
    labels = np.asarray(labels)
    return finalMatrix, labels
'''

def shuffleFunction(a, b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size // len(a)].reshape(a.shape)
    b2 = c[:, a.size // len(a):].reshape(b.shape)
    return a2, b2

def weightFunction(weightCondition, cols):
    if weightCondition is 'zeros':
        w = np.zeros(cols+1, dtype=np.int)
    if weightCondition is 'random':
        #w = np.zeros(cols + 1, dtype=np.int)
        w = np.ones(cols+1, dtype=np.int)*random.gauss(0, 0.1)
    if type(weightCondition) is not str:
        w = weightCondition
    return w

def simplePerceptron(matRix, labels, weightCondition, r):
    rows, cols = matRix.shape
    numUpdates = 0
    w = weightFunction(weightCondition, cols)
    for r_ind in range(rows):
        #Perceptron decision
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w,currExample))
        y_actual = labels[r_ind]
        #Decision
        if y_pred*y_actual <= 0:
           # Update
            w = w + r*y_actual*currExample
            numUpdates += 1
    return w, numUpdates

def marginPerceptron(matRix, labels, weightCondition, r, margin):
    rows, cols = matRix.shape
    numUpdates = 0
    w = weightFunction(weightCondition, cols)
    for r_ind in range(rows):
        #Perceptron decision
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w,currExample))
        y_actual = labels[r_ind]
        #Decision
        if y_pred*y_actual <= margin:
           # Update
            w = w + r*y_actual*currExample
            numUpdates += 1
    return w, numUpdates

def agrresiveMarginPerceptron(matRix, labels, weightCondition, r, margin):
    rows, cols = matRix.shape
    numUpdates = 0
    w = weightFunction(weightCondition, cols)
    for r_ind in range(rows):
        #Perceptron decision
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w,currExample))
        y_actual = labels[r_ind]
        #Decision
        if y_pred*y_actual <= margin:
            # Update
            n = (margin - y_pred*y_actual)/(np.dot(currExample,currExample)+1)
            w = w + n*y_actual*currExample
            numUpdates += 1
    return w, numUpdates

def accuracyFunc(matRix, labels, w):
    rows, cols = matRix.shape
    if (w.size-1)!= cols:
        num_extra_features = cols - (w.size - 1)
        for e in range(num_extra_features):
            w = np.append(w,[0])
    errorCount = 0
    for r_ind in range(rows):
        currExample = np.append([1], matRix[r_ind])
        y_pred = np.sign(np.dot(w, currExample))
        y_actual = labels[r_ind]
        #Decision
        if y_pred != y_actual:
            errorCount += 1
    accuracy = ((rows - errorCount)/rows)*100
    return accuracy

def algoFORepochs(numEpochs, train_filepath, test_filepath, algo_type, learningRate, margin = None):
    infoArray = [numEpochs, algo_type]
    trainMatrix, trainLabels = readFile(train_filepath)
    testMatrix, testLabels = readFile(test_filepath)
    tempTrainMatrix = copy.deepcopy(trainMatrix)
    tempTrainLabels = copy.deepcopy(trainLabels)

    for noYesShuffle in range(2):
        total_numUpdates = 0
        w = 0
        for e_ind in range(numEpochs):
            if noYesShuffle == 0:
                shuffleTYPE = 'no shuffle'
                trainingExMat = trainMatrix
                trainExLabels = trainLabels
            elif noYesShuffle == 1:
                shuffleTYPE = 'with shuffle'
                if e_ind == 0:
                    trainingExMat = trainMatrix
                    trainExLabels = trainLabels
                else:
                    #trainingExMat, trainExLabels = shuffle(tempTrainMatrix, tempTrainLabels, random_state=e_ind)
                    trainingExMat, trainExLabels = shuffleFunction(tempTrainMatrix, tempTrainLabels)
                    tempTrainMatrix = trainingExMat
                    tempTrainLabels = trainExLabels
            if e_ind == 0:
                weightCondition = 'random'
            else:
                weightCondition = w
            if algo_type == 'simple':
                w, numUpdates = simplePerceptron(trainingExMat, trainExLabels, weightCondition, learningRate)
            elif algo_type == 'margin':
                w, numUpdates = marginPerceptron(trainingExMat, trainExLabels, weightCondition, learningRate, margin)
            elif algo_type == 'aggressive':
                w, numUpdates = agrresiveMarginPerceptron(trainingExMat, trainExLabels, weightCondition, learningRate, margin)
            total_numUpdates += numUpdates

        acc_train = round(accuracyFunc(trainMatrix, trainLabels, w),2)
        acc_test = round(accuracyFunc(testMatrix, testLabels, w),2)
        infoArray.append(shuffleTYPE)
        infoArray.append(total_numUpdates)
        infoArray.append(acc_train)
        infoArray.append(acc_test)
    return infoArray
