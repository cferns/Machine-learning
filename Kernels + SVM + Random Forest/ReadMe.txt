# Environment 
os name: 7 centos, cade machine name: LAB1-23.

To run my code and replicate the experimental results in the write up:
1) Navigate to the folder titled CS6350_HW5
2) Run the script by entering the command:
./runme.sh
3)Done



Given below is the output I got when I ran my code:

3.1. Support Vector Machines:

3.1.1: solution
	Training accuracy: 93.2 %
	Test accuracy 91.4 %

3.1.2: solution
	gamma    C        Av accuracy %
	0.1    2         50.0
	0.1    0.125         56.45
	0.1    0.0625         55.65
	0.1    0.03125         54.8
	0.1    0.015625         53.85
	0.1    0.0078125         54.5
	0.001    2         50.0
	0.001    0.125         53.9
	0.001    0.0625         56.45
	0.001    0.03125         56.9
	0.001    0.015625         55.3
	0.001    0.0078125         54.55
	0.0001    2         52.55
	0.0001    0.125         52.2
	0.0001    0.0625         52.7
	0.0001    0.03125         55.8
	0.0001    0.015625         54.55
	0.0001    0.0078125         51.85
	Range of gamma(0):  [0.1, 0.001, 0.0001]
	Range of C:  [2, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
	Best gamma(0) :  0.001
	Best C:  0.03125
	Training accuracy: 61.1 %
	Test accuracy 57.67 %

3.1.3
	On the Handwritten data (Q 3.1.1):
		On training:
			Precision:  0.934782608696
			Recall:  0.941605839416
			F1 score:  0.938181818182
		On test:
			Precision:  0.918032786885
			Recall:  0.915032679739
			F1 score:  0.916530278232
	On the Madelon data (Q 3.1.2):
		On training:
			Precision:  0.603159851301
			Recall:  0.649
			F1 score:  0.625240847784
		On test:
			Precision:  0.573717948718
			Recall:  0.596666666667
			F1 score:  0.584967320261


Time taken to read Handwritten data:  4.60519180632
Time taken to read and (feature transform) Madelon data:  28.5535175957
3.2. Ensembles of decision trees:

3.2.1: solution
	Training accuracy: 99.4 %
	Test accuracy 93.59 %


3.2.2.a: solution
	For N =  10
		Training accuracy: 99.0 %
		Test accuracy 62.67 %
	For N =  30
		Training accuracy: 100.0 %
		Test accuracy 65.33 %
	For N =  100
		Training accuracy: 100.0 %
		Test accuracy 69.33 %

3.2.3.a: solution
	Best N =  100
	For Training:
		Accuracy:  100.0
		Precision:  1.0
		Recall:  1.0
		F1 score:  1.0
	For Test:
		Accuracy:  69.33
		Precision:  0.652631578947
		Recall:  0.826666666667
		F1 score:  0.729411764706

Time for code to run: 690.630500955


