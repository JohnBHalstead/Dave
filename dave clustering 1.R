#######################################################################################
### Clustering Dave Customers based on Spending Habits
#### John B. Halstead, Ph.D.
#### October 27 2019
#######################################################################################

### Libraries and Data

library(stats)
library(dplyr)
library(data.table)
library(zoo)
library(forecast)
library(e1071)
library(quantmod)
library(quadprog)
library(randomForest)
library(car)
library(ggplot2)
library(caret)
library(pROC)
library(reshape2)
library(useful)

# labeling the bank transaction data as bank

bank <- read.csv("~/data/bank_transaction.csv", header = TRUE, sep=",")
head(bank)
length(bank)
nrow(bank)

# crosstab amount to plaidCategory and bankAccountId

Spending <- setkey(setDT(bank), plaidCategory)[,list(total_amount=sum(amount)), by=list(bankAccountId, plaidCategory)]
head(Spending)
nrow(Spending)

spending <- acast(Spending, bankAccountId ~ plaidCategory) # bankAccountId becomes a row name
head(spending)
nrow(spending)
spending[is.na(spending)] <- 0 # replace NA with 0
head(spending)

#### Cluster of 10

set.seed(1357)
dave.cluster <- kmeans(x=spending, centers=10)
dave.cluster

plot(dave.cluster, data=spending) # from useful library

### finding the best cluster number

dave.Best <- FitKMeans(spending, max.clusters = 20, nstart = 25) # from useful library
dave.Best

PlotHartigan(dave.Best) # from useful library

### best cluster size is 3

set.seed(1357)
dave.cluster.best <- kmeans(x=spending, centers=3)
dave.cluster.best

plot(dave.cluster.best, data=spending) # from useful library

myClusters <- as.data.frame(dave.cluster.best$cluster)

write.csv(myClusters, file = "~/data/dave_clusters.csv")