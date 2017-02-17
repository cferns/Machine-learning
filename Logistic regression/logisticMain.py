import logisticFunctions
import timeit
import numpy as np
import matplotlib.pyplot as plt

readtimeSTART = timeit.default_timer()
train_filepath = './data/a5a.train'
test_filepath = './data/a5a.test'
table_filepath = './data/table2'
trainMatrix, trainLabels = logisticFunctions.readFile(train_filepath)
testMatrix, testLabels = logisticFunctions.readFile(test_filepath)
tableMatrix, tableLabels = logisticFunctions.readFile(table_filepath)
readtimeEND = timeit.default_timer()
#print 'Time to read the Data is: ', readtimeEND - readtimeSTART ,' secs'

#Cross validation
SigmaList = [1,50,100,125,150,175,200,225,250]
numberOfFolds = 10; CVepochs = 5; CVrate = 0.01
print '\nResults of ',numberOfFolds,'-fold Cross validation with ',CVepochs,' epochs and learning rate of ',CVrate,'are:'
store, bestSigma = logisticFunctions.crossValidation(trainMatrix, trainLabels, SigmaList, numberOfFolds, CVepochs, CVrate)

#Logistic Regression with Best Sigma
testEpochs =20; testRate = CVrate;
w, costFuncList = logisticFunctions.logisticRegression(trainMatrix, trainLabels, testEpochs, bestSigma, testRate)
trainPredictedLabels = logisticFunctions.prediction(trainMatrix, w)
testPredictedLabels = logisticFunctions.prediction(testMatrix, w)
trainAccuracy = logisticFunctions.accuracyMETRIC(trainLabels, trainPredictedLabels)
testAccuracy = logisticFunctions.accuracyMETRIC(testLabels, testPredictedLabels)
print '\nNumber of Epochs for final run: ', testEpochs
print 'Best Sigma through Cross Validation: ', bestSigma
print 'Accuracy on training data: ', trainAccuracy, ' %'
print 'Accuracy on test data: ', testAccuracy, ' %'

print '\nThe plot of \'Objective\' v/s. \'Number of epochs\' is plotted in a new window.'
plt.plot(range(1,testEpochs+1), costFuncList)
plt.xlabel('Number of epochs')
plt.ylabel('Objective')
plt.title('Plot of \'Objective\' v/s. \'Number of epochs')
plt.show()

codetimeEND = timeit.default_timer()
#print '\nTime taken for the whole code is ',codetimeEND - readtimeSTART ,' secs'

