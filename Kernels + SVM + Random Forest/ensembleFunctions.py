import svmFunctions
import numpy as np
import copy
import treeFunctions
from collections import Counter
import treeFunctions
import math
import random

def features_n_labels_matrix(featureMat, labelMat):
    finalmatrix = []
    for i in range(len(featureMat)):
        g = np.append(featureMat[i],labelMat[i])
        finalmatrix.append(g.tolist())
        npArray = np.array(finalmatrix)
    return npArray

def readDATA(matrixPATH, labelsPATH):
    matriX = np.array(svmFunctions.read_file(matrixPATH))
    labelS = np.array(svmFunctions.read_file(labelsPATH))
    labelS = [item for sublist in labelS for item in sublist]
    finalmatrix = features_n_labels_matrix(matriX, labelS)
    return finalmatrix, labelS

def readDATA_and_transform(trainMATpath, trainLABELSpath, testMATpath, testLABELSpath):
    trainMATRIX = np.array(svmFunctions.read_file(trainMATpath))
    testMATRIX = np.array(svmFunctions.read_file(testMATpath))
    stdConstant = 1
    trainMATRIX_transformed, testMATRIX_transformed = featureTransformation(trainMATRIX, testMATRIX, stdConstant)
    trainLABELS = np.array(svmFunctions.read_file(trainLABELSpath))
    trainLABELS = [item for sublist in trainLABELS for item in sublist]
    testLABELS = np.array(svmFunctions.read_file(testLABELSpath))
    testLABELS = [item for sublist in testLABELS for item in sublist]
    finalTRAINmatrix = features_n_labels_matrix(trainMATRIX_transformed, trainLABELS)
    finalTESTmatrix = features_n_labels_matrix(testMATRIX_transformed, testLABELS)
    return finalTRAINmatrix, trainLABELS, finalTESTmatrix, testLABELS

def featureTransformation(trainMatrix, testMatrix, stdConstant):
    #'below code is for outliers'
    numTrainRows = len(trainMatrix)
    numTrainFeatures = len(trainMatrix[0])
    numTestRows = len(testMatrix)
    for f in range(numTrainFeatures):
        stDev = np.std(trainMatrix[:, f])
        meaN = np.mean(trainMatrix[:, f])
        upLimit = meaN + stDev*stdConstant
        lowLimit = meaN - stDev*stdConstant
        numBuckets = math.ceil(math.sqrt(stDev))
        if numBuckets == 0:
            bucketSize = 1
        else:
            bucketSize = math.ceil(upLimit / numBuckets)
        for r in range(numTrainRows):
            if trainMatrix[r][f] > upLimit:
                trainMatrix[r][f] = upLimit
            #if trainMatrix[r][f] < lowLimit:
             #   trainMatrix[r][f] = lowLimit
            trainMatrix[r][f] = math.ceil(trainMatrix[r][f] / bucketSize)
        for rt in range(numTestRows):
            if testMatrix[rt][f] > upLimit:
                testMatrix[rt][f] = upLimit
            #if testMatrix[rt][f] < lowLimit:
            #    testMatrix[rt][f] = lowLimit
            testMatrix[rt][f] = math.ceil(testMatrix[rt][f] / bucketSize)

    return trainMatrix, testMatrix

def treeEnsembles(train_MATRIX, train_LABELS, test_MATRIX, N, mPercent):
    ensembleDataset_train_TRANS = []
    ensembleDataset_test_TRANS = []
    numSamples = int(math.ceil(len(train_MATRIX) * mPercent))
    for numTrees in range(N):
        Sdata = np.array([random.choice(train_MATRIX) for _ in range(numSamples)])

        predicted_train_list = []
        predicted_test_list = []
        #
        num_TrainingFEATURES = len(Sdata[0])-1
        depth_of_tree = 0; limiting_depth = 10000;
        to_be_visited = range(num_TrainingFEATURES)
        major_trainLABEL = (Counter(train_LABELS)).most_common(1)[0][0]
        tree = treeFunctions.create_ID3_tree(Sdata, num_TrainingFEATURES, depth_of_tree, to_be_visited,
                                                major_trainLABEL, limiting_depth)
        list_of_labels = list(Counter(train_LABELS))
        predicted_train_list = treeFunctions.predictFUNCTION(tree, train_MATRIX, list_of_labels, major_trainLABEL)
        predicted_test_list = treeFunctions.predictFUNCTION(tree, test_MATRIX, list_of_labels, major_trainLABEL)
        ensembleDataset_train_TRANS.append(predicted_train_list)
        ensembleDataset_test_TRANS.append(predicted_test_list)

    ensembleDataset_train = np.transpose(np.array(ensembleDataset_train_TRANS))
    ensembleDataset_test = np.transpose(np.array(ensembleDataset_test_TRANS))

    return ensembleDataset_train, ensembleDataset_test