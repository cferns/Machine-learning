import percepFunctions
import timeit

train_filepath = './data/a5a.train'
test_filepath = './data/a5a.test'
table_filepath = './data/table2'
trainMatrix, trainLabels = percepFunctions.readFile(train_filepath)
testMatrix, testLabels = percepFunctions.readFile(test_filepath)
tableMatrix, tableLabels = percepFunctions.readFile(table_filepath)
learningRate = 1
margin = 1e-06

start_time = timeit.default_timer()

print '\n------------------ Q.3.3.1 Solution: ------------------'
weightCondition = 'zeros'
learningRate = 1
w, numUpdates = percepFunctions.simplePerceptron(tableMatrix, tableLabels, weightCondition, learningRate)
print 'The weight vector is w^t with bias as the first term: ', w
print 'Number of mistakes made: ', numUpdates


print '\n\n\n------------------ Q.3.3.2 Solution: ------------------'
weightCondition = 'random'

w_simpleP, numUpdates_simpleP = percepFunctions.simplePerceptron(trainMatrix, trainLabels, weightCondition, learningRate)
w_marginP, numUpdates_marginP = percepFunctions.marginPerceptron(trainMatrix, trainLabels, weightCondition, learningRate, margin)
acc_simpleP_train = round(float(percepFunctions.accuracyFunc(trainMatrix, trainLabels, w_simpleP)),2)
acc_simpleP_test = round(float(percepFunctions.accuracyFunc(testMatrix, testLabels, w_simpleP)),2)
acc_marginP_train = round(float(percepFunctions.accuracyFunc(trainMatrix, trainLabels, w_marginP)),2)
acc_marginP_test = round(float(percepFunctions.accuracyFunc(testMatrix, testLabels, w_marginP)),2)

print '      ----            Perceptron   Margin_Perceptron'
print 'num of Updates :       %s        %s' %(numUpdates_simpleP,numUpdates_marginP)
print 'training accuracy %% :  %s       %s' %(acc_simpleP_train,acc_marginP_train)
print 'test accuracy %% :      %s       %s' %(acc_simpleP_test,acc_marginP_test)



print '\n\n\n------------------ Q.3.3.3 Solution: ------------------'
simple_3epochs = percepFunctions.algoFORepochs(3, train_filepath, test_filepath, 'simple', learningRate, margin)
print 'For %s epochs on the %s Perceptron, the results were following: ' %(simple_3epochs[0],simple_3epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(simple_3epochs[3],simple_3epochs[3+4])
print 'training accuracy %% :  %s       %s' %(simple_3epochs[4],simple_3epochs[4+4])
print 'test accuracy %% :      %s       %s' %(simple_3epochs[5],simple_3epochs[5+4])

#5 epochs ; simple ; w and w/o shuffle
simple_5epochs = percepFunctions.algoFORepochs(5, train_filepath, test_filepath, 'simple', learningRate, margin)
print '\nFor %s epochs on the %s Perceptron, the results were following: ' %(simple_5epochs[0],simple_5epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(simple_5epochs[3],simple_5epochs[3+4])
print 'training accuracy %% :  %s       %s' %(simple_5epochs[4],simple_5epochs[4+4])
print 'test accuracy %% :      %s       %s' %(simple_5epochs[5],simple_5epochs[5+4])

#3 epochs ; margin ; w and w/o shuffle
margin_3epochs = percepFunctions.algoFORepochs(3, train_filepath, test_filepath, 'margin', learningRate, margin)
print '\nFor %s epochs on the %s Perceptron, the results were following: ' %(margin_3epochs[0],margin_3epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(margin_3epochs[3],margin_3epochs[3+4])
print 'training accuracy %% :  %s       %s' %(margin_3epochs[4],margin_3epochs[4+4])
print 'test accuracy %% :      %s       %s' %(margin_3epochs[5],margin_3epochs[5+4])

#5 epochs ; margin ; w and w/o shuffle
margin_5epochs = percepFunctions.algoFORepochs(5, train_filepath, test_filepath, 'margin', learningRate, margin)
print '\nFor %s epochs on the %s Perceptron, the results were following: ' %(margin_5epochs[0],margin_5epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(margin_5epochs[3],margin_5epochs[3+4])
print 'training accuracy %% :  %s       %s' %(margin_5epochs[4],margin_5epochs[4+4])
print 'test accuracy %% :      %s       %s' %(margin_5epochs[5],margin_5epochs[5+4])



print '\n\n\n------------------ Q.3.3.4 Solution: ------------------'
#3 epochs ; aggressive ; w and w/o shuffle
aggressive_3epochs = percepFunctions.algoFORepochs(3, train_filepath, test_filepath, 'aggressive', learningRate, margin)
print 'For %s epochs on the %s Perceptron, the results were following: ' %(aggressive_3epochs[0],aggressive_3epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(aggressive_3epochs[3],aggressive_3epochs[3+4])
print 'training accuracy %% :  %s       %s' %(aggressive_3epochs[4],aggressive_3epochs[4+4])
print 'test accuracy %% :      %s       %s' %(aggressive_3epochs[5],aggressive_3epochs[5+4])

#5 epochs ; aggressive ; w and w/o shuffle
aggressive_5epochs = percepFunctions.algoFORepochs(5, train_filepath, test_filepath, 'aggressive', learningRate, margin)
print '\nFor %s epochs on the %s Perceptron, the results were following: ' %(aggressive_5epochs[0],aggressive_5epochs[1])
print '      ----            noShuffle   with_Shuffle'
print 'num of Updates :       %s        %s' %(aggressive_5epochs[3],aggressive_5epochs[3+4])
print 'training accuracy %% :  %s       %s' %(aggressive_5epochs[4],aggressive_5epochs[4+4])
print 'test accuracy %% :      %s       %s' %(aggressive_5epochs[5],aggressive_5epochs[5+4])


stop_time = timeit.default_timer()
print '\nTime taken for the code to run: ', stop_time - start_time