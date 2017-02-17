# Environment 
os name: 7 centos, cade machine name: LAB1-23.

To run my code and replicate the experimental results in the write up:
1) Navigate to the folder titled CS6350_HW6
2) Run the script by entering the command:
./runme.sh
3)Done


Given below is the 'Copy of the output printed in the terminal' after running 'runme.sh':


[cfernand@lab1-23 CS6350_HW6]$ ./runme.sh

Results of  10 -fold Cross validation with  5  epochs and learning rate of  0.01 are:
	sigma    Average accuracy(%)
	  1      75.586
	  50      84.019
	  100      84.316
	  125      84.051
	  150      84.393
	  175      84.126
	  200      84.390
	  225      84.191
	  250      84.003

Number of Epochs for final run:  20
Best Sigma through Cross Validation:  150
Accuracy on training data:  85.27  %
Accuracy on test data:  84.64  %

The plot of 'Objective' v/s. 'Number of epochs' is plotted in a new window.

Process finished with exit code 0