from __future__ import division
import ensembleFunctions
import svmFunctions
import timeit

s1 = timeit.default_timer()
start = timeit.default_timer()

hand_train_wholeMAT, hand_train_labels = ensembleFunctions.readDATA('./data/handwriting/train.data','./data/handwriting/train.labels')
hand_test_wholeMAT, hand_test_labels = ensembleFunctions.readDATA('./data/handwriting/test.data','./data/handwriting/test.labels')
e1 = timeit.default_timer()
print 'Time taken to read Handwritten data: ', e1-s1
s2 = timeit.default_timer()
made_train_wholeMAT, made_train_labels, made_test_wholeMAT, made_test_labels = ensembleFunctions.readDATA_and_transform('./data/madelon/madelon_train.data','./data/madelon/madelon_train.labels','./data/madelon/madelon_test.data','./data/madelon/madelon_test.labels')
e2 = timeit.default_timer()
print 'Time taken to read and (feature transform) Madelon data: ', e2-s2


print '\n3.2. Ensembles of decision trees:'
print '\n3.2.1: solution'
N=5; mPercent321 = 1#mPercent is the sampling percentage
ensData_train_321, ensData_test_321 = ensembleFunctions.treeEnsembles(hand_train_wholeMAT, hand_train_labels, hand_test_wholeMAT, N, mPercent321)
epochs_321 = 2; C_321 = 1;rate_321 = 0.01
w_321 = svmFunctions.SVM(ensData_train_321, hand_train_labels, epochs_321, C_321, rate_321)
pred_train_list_321 = svmFunctions.prediction(ensData_train_321, hand_train_labels, w_321)
pred_test_list_321 = svmFunctions.prediction(ensData_test_321, hand_test_labels, w_321)
acc_train_321 = svmFunctions.accuracyMETRIC(hand_train_labels, pred_train_list_321)
acc_test_321 = svmFunctions.accuracyMETRIC(hand_test_labels, pred_test_list_321)
print '\tTraining accuracy:', acc_train_321,'%'
print '\tTest accuracy', acc_test_321,'%'

print '\n3.2.2.a: solution'
Nlist = [10,30,100]
epochs_322 = 1; C_322 = 1;rate_322 = 0.01; mPercent322 = 2000/2000
bestTestAccuracy = 0
for N in Nlist:
    ensData_train_322, ensData_test_322 = ensembleFunctions.treeEnsembles(made_train_wholeMAT, made_train_labels, made_test_wholeMAT, N, mPercent322)
    w_322 = svmFunctions.SVM(ensData_train_322, made_train_labels, epochs_322, C_322, rate_322)
    pred_train_list_322 = svmFunctions.prediction(ensData_train_322, made_train_labels, w_322)
    pred_test_list_322 = svmFunctions.prediction(ensData_test_322, made_test_labels, w_322)
    acc_train_322 = svmFunctions.accuracyMETRIC(made_train_labels, pred_train_list_322)
    acc_test_322 = svmFunctions.accuracyMETRIC(made_test_labels, pred_test_list_322)
    print '\tFor N = ', N
    print '\t\tTraining accuracy:', acc_train_322, '%'
    print '\t\tTest accuracy', acc_test_322, '%'
    if acc_test_322 > bestTestAccuracy:
        bestN = N
        pred_train_list_323 = pred_train_list_322
        pred_test_list_323 = pred_test_list_322

print '\n3.2.3.a: solution'
print '\tBest N = ', bestN
print '\tFor Training:'
acc_train_323 = svmFunctions.accuracyMETRIC(made_train_labels, pred_train_list_323)
print '\t\tAccuracy: ', acc_train_323
precision_323_train, recall_323_train, f1_323_train = svmFunctions.f1METRIC(made_train_labels, pred_train_list_323)
print '\t\tPrecision: ', precision_323_train
print '\t\tRecall: ', recall_323_train
print '\t\tF1 score: ', f1_323_train
#
print '\tFor Test:'
acc_test_323 = svmFunctions.accuracyMETRIC(made_test_labels, pred_test_list_323)
print '\t\tAccuracy: ',acc_test_323
precision_323_test, recall_323_test, f1_323_test = svmFunctions.f1METRIC(made_test_labels, pred_test_list_323)
print '\t\tPrecision: ', precision_323_test
print '\t\tRecall: ', recall_323_test
print '\t\tF1 score: ', f1_323_test

end = timeit.default_timer()
print end - start
