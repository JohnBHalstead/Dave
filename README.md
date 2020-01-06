# Dave
R Code written for Dave.com

There are two R files. One file builds, tests, and optimizes a three day bank account signal. The other file determines the creates the optimal clusters based on people's bank accounts who use Dave.com as a service.

The classification file manipulates bank account data to create a virtual bank account. Many R skills are used, including SQL and programs. The manipulated and clean data are randomly divided into training and test data. Two Machine Learning algorthims are applied to the data (Random Forest and SVM). Model accuracy is measured by AUC and ROC characteristics. The RandomForest was selected and further optimized by tweaking the number of trees in the forest.

The clustering file used KNN to cluster the bank accounts. FitKmeans aglorithm was utilized to select the cluster numbers.
