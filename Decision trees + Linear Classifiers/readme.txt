# Environment 
os name: 7 centos, cade machine name: LAB1-23.

To run my code and replicate the experimental results in the write up:
1) Navigate to the folder titled CS6350_HW1
2) Run the script by entering the command:
./runme.sh
3)Done


Given below is the 'Copy of the output printed in the terminal' after running 'runme.sh':


[cfernand@lab1-23 CS6350_HW1]$ ./runme.sh
code is running

 --------------- Setting A solutions: --------------- 

A.1.a : Answer in Report
A.1.b : Error of decision tree on the SettingA/training.data file: 0 %
A.1.c : Error of decision tree on the SettingA/test.data file: 0 %
A.1.d : Maximum depth of decision tree: 3
A.2.a : Average cross-validation accuracy and standard deviation for each depth: 
depth       | accuracy |            std_deviation
1        97.6452119309        5.12633590812
2        98.1423338566        4.1538677761
3        98.1423338566        4.1538677761
4        98.1423338566        4.1538677761
5        98.1423338566        4.1538677761
10        98.1423338566        4.1538677761
15        98.1423338566        4.1538677761
20        98.1423338566        4.1538677761
A.2.b : Accuracy of decision tree on the SettingA/test.data file: (with best depth = 2) is: 99.780461
Time taken for Setting A code to run:  17.0235571861

 --------------- Setting B solutions: --------------- 

B.1.a : Error of decision tree on the SettingB/training.data : 0.000 %
B.1.b : Error of decision tree on the SettingB/test.data file : 6.476 %
B.1.c : Error of decision tree on the SettingA/training.data file : 0.052  %
B.1.d : Error of decision tree on the SettingA/test.data file : 0.220 %
B.1.e : Maximum depth of decision tree : 9
B.2.a : Cross-validation accuracy and standard deviation for each depth is below. Best depth : 4
depth       | accuracy |            std_deviation
1        60.5965463108        31.5964372829
2        92.883307169        4.29700533397
3        90.2145473574        9.81744114596
4        93.2496075353        2.15629490089
5        92.4908424908        2.83940050218
10        92.2291993721        2.89715855285
15        92.2291993721        2.89715855285
20        92.2291993721        2.89715855285
B.2.b : Accuracy of decision tree on the SettingB/test.data file : 94.182  %
Time taken for Setting B code to run:  64.6125388145

 --------------- Setting C solutions: --------------- 

C.1 : Answer in report : 
C.2 : Accuracy for each method and the standard deviation : 
depth     | m1-accuracy     | m1-std_deviation     | m2-accuracy     | m2-std_deviation     | m3-accuracy     | m3-std_deviation
1        73.5579420592        25.6746675731        73.5579420592        25.6746675731        73.5579420592        25.6746675731
2        97.8860347492        3.84691797872        97.8860347492        3.84691797872        97.8860347492        3.84691797872
3        98.8231902067        1.82042752258        98.8231902067        1.82042752258        98.8231902067        1.82042752258
4        99.1730981257        1.84900880168        99.1730981257        1.84900880168        99.1730981257        1.84900880168
5        99.1730981257        1.84900880168        99.1730981257        1.84900880168        99.1730981257        1.84900880168
10        99.1730981257        1.84900880168        99.1730981257        1.84900880168        99.1730981257        1.84900880168
15        99.1730981257        1.84900880168        99.1730981257        1.84900880168        99.1730981257        1.84900880168
20        99.1730981257        1.84900880168        99.1730981257        1.84900880168        99.1730981257        1.84900880168
C.3 : Using the best method, Accuracy of tree on SettingC/test.data : 100.000 %
Time taken for Setting C code to run:  136.272800922

Time taken for whole code to run:  217.908912182

