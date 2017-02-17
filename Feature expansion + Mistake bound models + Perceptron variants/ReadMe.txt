# Environment 
os name: 7 centos, cade machine name: LAB1-23.

To run my code and replicate the experimental results in the write up:
1) Navigate to the folder titled CS6350_HW2
2) Run the script by entering the command:
./runme.sh
3)Done




Given below is the 'Copy of the output printed in the terminal, after running 'runme.sh':


------------------ Q.3.3.1 Solution: ------------------
The weight vector is w^t with bias as the first term:  [ 0.  0.  1.  0. -1.  2.]
Number of mistakes made:  4



------------------ Q.3.3.2 Solution: ------------------
      ----            Perceptron   Margin_Perceptron
num of Updates :       1373        1389
training accuracy % :  81.35       82.02
test accuracy % :      81.09       81.47



------------------ Q.3.3.3 Solution: ------------------
For 3 epochs on the simple Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       4078        4056
training accuracy % :  79.83       82.46
test accuracy % :      78.89       81.34

For 5 epochs on the simple Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       6752        6736
training accuracy % :  83.29       81.26
test accuracy % :      82.79       80.14

For 3 epochs on the margin Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       4054        4097
training accuracy % :  79.62       82.85
test accuracy % :      79.4       81.79

For 5 epochs on the margin Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       6747        6696
training accuracy % :  80.23       80.87
test accuracy % :      79.78       80.46



------------------ Q.3.3.4 Solution: ------------------
For 3 epochs on the aggressive Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       4072        4102
training accuracy % :  80.57       83.26
test accuracy % :      79.77       82.47

For 5 epochs on the aggressive Perceptron, the results were following: 
      ----            noShuffle   with_Shuffle
num of Updates :       6720        6722
training accuracy % :  80.34       81.04
test accuracy % :      79.57       81.34

Time taken for Q.3. code to run:  11.7175716656

Process finished with exit code 0