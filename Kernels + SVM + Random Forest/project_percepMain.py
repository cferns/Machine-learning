import percepFunctions
import projectFunctions
from collections import Counter
import timeit
import crossValidationExperiments
import numpy as np
import scipy
import operator

#Read original DATA
train_filepath = './data/data-splits/data.train'
train_matrix, train_actual_labels = percepFunctions.readFile_multi(train_filepath,'w')
eval_anon_filepath = './data/data-splits/data.eval.anon'
eval_matrix, eval_actual_labels = percepFunctions.readFile_multi(eval_anon_filepath,'w')
test_filepath = './data/data-splits/data.test'
test_matrix, test_actual_labels = percepFunctions.readFile_multi(test_filepath,'w')
e1 = timeit.default_timer()

trainMatrix=train_matrix
testMatrix=test_matrix
evalMatrix=eval_matrix

#Featurize the data
#trainMatrix, testMatrix, evalMatrix  = projectFunctions.featureTransformation(train_matrix, test_matrix, eval_matrix,1)

#Simple-Weighted Perceptron w Epochs

learningRate = 0.675
margin = 0
maxEpochs = 1
bestLIST = percepFunctions.algoFORepochs(maxEpochs, trainMatrix, train_actual_labels, testMatrix, test_actual_labels, evalMatrix, eval_actual_labels, 'simple', learningRate, margin)
for i in range(1,7):
  print bestLIST[i]
for i in range(8,14):
  print bestLIST[i]
#Write leaderboard file
solution_filename = raw_input('Enter x__x in ./solutions_log/x__x_solutions.csv: ')
w_val = bestLIST[7]
eval_pred_labels = projectFunctions.prediction_Perceptron(eval_matrix, eval_actual_labels, w_val)
projectFunctions.write_solutions('perceptron',eval_pred_labels,'./solutions_log/solutions/'+solution_filename+'.solutions.csv')


'''
filepath = './solutions_log/simple_50e_infolist.txt'
thefile = open('./solutions_log/infoARRAY/'+solution_filename+'.infoARRAY.csv', 'w')
for item in simple_weighted_info:
  thefile.write("%s\n" % item)
thefile.close()
'''
#end_time = timeit.default_timer()


#Simple or Weighted Perceptron
'''
#print '\n------------------ Simple Perceptron: ------------------'
#w_train, numUpdates_simpleP = percepFunctions.simplePerceptron(train_matrix, train_actual_labels, 'zeros', 0.85)
#print '\n------------------ Weighted Perceptron: ------------------'
#w_train, numUpdates_simpleP = percepFunctions.weightedPerceptron(train_matrix, train_actual_labels, 'zeros', 0.675)
#print '\n------------------ Aggressive Perceptron: ------------------'
w_train, numUpdates_simpleP = percepFunctions.WeightedAgrresiveMarginPerceptron(train_matrix, train_actual_labels, 'random', 0.675, 0)
#Prediction on train
train_pred_labels = projectFunctions.prediction_Perceptron(train_matrix, train_actual_labels, w_train)
train_accuracy = projectFunctions.accuracyMETRIC(train_actual_labels, train_pred_labels)
#Prediction on test
test_pred_labels = projectFunctions.prediction_Perceptron(test_matrix, test_actual_labels, w_train)
test_accuracy = projectFunctions.accuracyMETRIC(test_actual_labels, test_pred_labels)
#Prediction on eval.anon
eval_pred_labels = projectFunctions.prediction_Perceptron(eval_matrix, eval_actual_labels, w_train)
eval_accuracy = projectFunctions.accuracyMETRIC(eval_actual_labels, eval_pred_labels)

print 'Accuracy : train: ', train_accuracy
print 'Accuracy : test: ', test_accuracy
print 'Accuracy : eval.anon: ', eval_accuracy

#Write leaderboard file
#projectFunctions.write_solutions(eval_pred_labels,"simplePerceptron.rev1.csv")
'''

#Margin Perceptron
'''
print '\n------------------ Margin Perceptron: ------------------'
learningRate = 1
margin = 1
#Training
w_train, numUpdates_marginP = percepFunctions.marginPerceptron(train_matrix, train_actual_labels, 'zeros', learningRate, margin)
#Prediction on test
train_pred_labels = projectFunctions.prediction_Perceptron(train_matrix, train_actual_labels, w_train)
train_accuracy = projectFunctions.accuracyMETRIC(train_actual_labels, train_pred_labels)
print 'Accuracy : train: ', train_accuracy
#Prediction on test
test_pred_labels = projectFunctions.prediction_Perceptron(test_matrix, test_actual_labels, w_train)
test_accuracy = projectFunctions.accuracyMETRIC(test_actual_labels, test_pred_labels)
print 'Accuracy : test: ', test_accuracy
#Prediction on eval.anon
eval_pred_labels = projectFunctions.prediction_Perceptron(eval_matrix, eval_actual_labels, w_train)
eval_accuracy = projectFunctions.accuracyMETRIC(eval_actual_labels, eval_pred_labels)
print 'Accuracy : eval.anon: ', eval_accuracy
#Write leaderboard file
projectFunctions.write_solutions(eval_pred_labels,"solutions.csv")
'''

#cross validation experiment: on rates
'''
delT = 0.0125/2
st = 0.7-0.05+0.0125
ed = 0.6875-0.0125
#tempRateRange = np.arange(st,ed,delT)
tempRateRange = np.arange(0.1,1,0.1)
rateRange = tempRateRange.tolist()
#rateRange = [0,0.00001,0.0001,0.001,0.01,0.1,1]
num_folds = 10
best_f1, best_rate, av_f1_list = crossValidationExperiments.simplePerceptron_cvRate(train_matrix, train_actual_labels, rateRange, num_folds)
#best_accuracy, best_rate, av_accuracy_list = crossValidationExperiments.marginPerceptron_cvRate(train_matrix, train_actual_labels, rateRange, num_folds)
print 'best_f1',best_f1
print 'best_rate',best_rate
print 'av_f1_list',av_f1_list
'''

#cv weights
'''
num_folds = 10
projectFunctions.split_data(train_matrix, train_actual_labels, num_folds)
#cross validation experiment
cvStart = timeit.default_timer
weightRange = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1',]
num_folds = 5
best_accuracy, best_weight = crossValidationExperiments.simplePerceptron_cvWeight(train_matrix, train_actual_labels, weightRange, num_folds)
cvStop = timeit.default_timer
'''
