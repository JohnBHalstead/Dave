####################################################################################
#### Time Series Analysis for Dave's Bank
#### John B. Halstead, Ph.D.
#### October 16 2019
####################################################################################

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

# labeling the bank transaction data as bank

bank <- read.csv("~/data/bank_transaction.csv", header = TRUE, sep=",")
head(bank)
length(bank)
nrow(bank)
 

# labeling the balance log as blog... just go with it

blog <- read.csv("~/data/balance_log.csv", header = TRUE, sep=",")
head(blog)
length(blog)
nrow(blog)

### Spending habits

# crosstab amount to plaidCategory

crosstab.amount <- setkey(setDT(bank), plaidCategory)[,list(total_amount=sum(amount)), by=list(plaidCategory)]
crosstab.amount

# not legible so must print to file and reproduce elsewhere plot(crosstab.amount)
write.csv(crosstab.amount, file = "~/data/crosstab_data.csv")

### bank account plots

summary(blog)

hist(blog$available,
     main="Available Balance Histogram",
     xlab="Available Balance in USD",
     xlim=c(-20000,20000),
     col="darkmagenta"
)

hist(blog$current,
     main="Current Balance Histogram",
     xlab="Current Balance in USD",
     xlim=c(-20000,20000),
     col="lightsalmon"
)

#### Building a signal for a classification model

### How much warning time is needed?
### by observation once a new account is added the time prior stacks up, yet the average time between the
### transaction data and the date created averages less then 1 day. However, weekends (or three days) occur
### We are going to build a three day signal to provide early warning

## Order by bank account and transaction date 

bank <- bank[order(bank$bankAccountId, bank$transactionDate),]
head(bank)

## apply SQL into R group by bankAccountid and transactionDate to sum amount
## uses package dplyr

newbank <- bank %>%
        select(bankAccountId, amount, transactionDate) %>%
        group_by(bankAccountId, transactionDate) %>%
        summarise(amount = sum(amount))

head(newbank)
nrow(newbank)

## convert the tibble back to data frame

newbank <- as.data.frame(newbank)
head(newbank)
tail(newbank)

## adding a cumlative sum function column dependent upon bank account
## this creates a virtual bank account balance

newbank$AccountBalance <- ave(newbank$amount, newbank$bankAccountId, FUN=cumsum)
head(newbank)

## let's create a 3 day signal

#sign assignment

newbank$signal <- with(newbank, sign(newbank$AccountBalance))
head(newbank)

#relabel signals for classification tasks

newbank$signal <- with(newbank, newbank$signal*-1)
newbank$signal[newbank$signal==0] <- 1
head(newbank)

#shifting signal rows up 3 to create a warning time

shift <- function(x, n){
        c(x[-(seq(n))], rep(NA, n))
}

newbank$signal <- shift(newbank$signal, 3)
head(newbank)
tail(newbank)

## Current state of newbank signals are ideal of SVM with -1 and 1 and three NA on the bottom of the data
## The number 1 is a signal, for SVM we retain the -1 and remove the NA
## For RandomForest, we convert the -1 to zero and remove the NA

## SVM data

newbank.svm <- newbank
newbank.svm$signal[is.na(newbank.svm$signal)] <- -1
head(newbank.svm)
tail(newbank.svm)


## converting -1 and NA to 0 for most other classification models including RandomForest

newbank$signal[newbank$signal<0] <- 0
newbank$signal[is.na(newbank$signal)] <- 0
head(newbank)
tail(newbank)

##########################################################################################
## create data sets for classification models
##########################################################################################

### randomize training and test data
## 2/3 and 1/3 train and test set rule for 10K random samples

set.seed(357)
rs <- sample(1:nrow(newbank), size = 10000, replace=FALSE)
head(rs)

## for RandomForest data
ModelData <- newbank[rs,]
head(ModelData)

## for SVM data

ModelData.svm <- newbank.svm[rs,]
head(ModelData.svm)

## training and test data for RF

set.seed(159)
sample_t <- sample(1:nrow(ModelData), size = (nrow(ModelData)*(2/3)), replace=FALSE)

train <- ModelData[sample_t,]
head(train)

test <- ModelData[-sample_t,]
head(test)

## training and test data for svm

set.seed(379)
sample_t <- sample(1:nrow(ModelData.svm), size = (nrow(ModelData.svm)*(2/3)), replace=FALSE)

train.svm <- ModelData.svm[sample_t,]
head(train.svm)

test.svm <- ModelData.svm[-sample_t,]
head(test.svm)

###########################################################################################
### classification Models
###########################################################################################

#### Random Forest model

set.seed(73)
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, importance=TRUE, proximity=TRUE, type = prob)
print(dave.rf)
round(importance(dave.rf), 2)
varImpPlot(dave.rf, type = 1)

y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
sse <- (y.hat-y1)^2
mse <- sum(sse)/length(y1)

### RF ROC

roc(y1 ~ y.hat)

roc1 <- roc(y1,
            y.hat, percent=TRUE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=FALSE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)

coords(roc1, "best", ret=c("threshold", "specificity", "1-npv"))
ci(roc1)

sens.ci <- ci.se(roc1, specificities=seq(0, 100, 5))
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")

#### SVM

dave.svm <- svm(signal ~ amount + AccountBalance, data=train.svm)
print(dave.svm)

y.hat.svm <- predict(dave.svm, newdata = test.svm)
y1.svm <- test.svm$signal
sse.svm <- (y.hat.svm-y1.svm)^2
mse.svm <- sum(sse.svm)/length(y1.svm)

roc(y1.svm ~ y.hat.svm)

roc2 <- roc(y1.svm,
            y.hat.svm, percent=TRUE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=FALSE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)

coords(roc2, "best", ret=c("threshold", "specificity", "1-npv"))
ci(roc2)

sens.ci <- ci.se(roc2, specificities=seq(0, 100, 5))
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")


#### Comparing RF to sVM

roc.test(roc1, roc2, reuse.auc=FALSE)

# With modified bootstrap parameters
roc.test(roc1, roc2, reuse.auc=FALSE, partial.auc=c(100, 90),
         partial.auc.correct=TRUE, boot.n=1000, boot.stratified=FALSE)

#### Random Forest beats the SVM

#### Tweet the RandomForest model for more generalization/accuracy

set.seed(246)
rf.trees <- NULL
rf.auc <- NULL
i <- NULL
j <- NULL

for (i in 20){
        j <- i*50
        rf.trees <- rf.trees + j
        dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=j,importance=TRUE, proximity=TRUE, type = prob)
        y.hat <- predict(dave.rf, newdata = test)
        y1 <- test$signal
        newby <- roc(y1 ~ y.hat)
        new.roc <- newby$auc
        rf.auc <- rf.auc + new.roc
}

### manually roc variable not coming out by itself, disrupting the loop

## 100 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=100,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 200 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=200,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 300 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=300,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 400 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=400,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 500 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=500,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 600 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=600,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 700 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=700,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 800 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=800,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 900 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=900,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

## 1000 trees
dave.rf <- randomForest(signal ~ amount + AccountBalance, data=train, ntree=1000,importance=TRUE, proximity=TRUE, type = prob)
y.hat <- predict(dave.rf, newdata = test)
y1 <- test$signal
roc(y1 ~ y.hat)

##########################################################################################################
### Best of the best is a RandomForest with 300 trees in the forest
##########################################################################################################

#### Random Forest model for 300 trees

set.seed(73)
dave.rf.1 <- randomForest(signal ~ amount + AccountBalance, data=train, ntree = 300, importance=TRUE, proximity=TRUE, type = prob)
print(dave.rf.1)
round(importance(dave.rf.1), 2)
varImpPlot(dave.rf.1, type = 1)

y.hat.1 <- predict(dave.rf.1, newdata = test)
y1.1 <- test$signal
sse <- (y.hat.1-y1.1)^2
mse <- sum(sse)/length(y1.1)

### RF ROC

roc(y1.1 ~ y.hat.1)

roc3 <- roc(y1,
            y.hat.1, percent=TRUE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=FALSE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)

coords(roc3, "best", ret=c("threshold", "specificity", "1-npv"))
ci(roc3)

sens.ci <- ci.se(roc3, specificities=seq(0, 100, 5))
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")


###########################################################################################
### Translating to Python
###########################################################################################

### Data tansformations can be handled in a similar way by the python package pandas and the sql aspect
### by pandasql

### Both RandomForest and SVM are handled by the Python package scikit-learn

### Scikit defaults the amount of trees in a RandomForest to 50 trees. To run Python, modify the Python Random
### Forest to 500 trees.

### Example of Python

# Required Python Packages

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# Data

# INPUT_PATH = "../inputs/balance_log.csv"
# OUTPUT_PATH = "../inputs/models.csv"


# def data_file_to_csv():

# Headers 

# headers = [ "bankAccountId", "displayName", "amount", "plaidCategory", "transactionDate", "created"]

# def read_data(path):
#        """
#   Read the data into pandas dataframe
#    :param path:
#    :return:
#    """
# data = pd.read_csv(path)
# return data


# def get_headers(dataset):
#        """
#    dataset headers
#    :param dataset:
#    :return:
#    """
# return dataset.columns.values


# def add_headers(dataset, headers):
#        """
#    Add the headers to the dataset
#    :param dataset:
#    :param headers:
#    :return:
#    """
# dataset.columns = headers
# return dataset


# def data_file_to_csv():
#        """
 
#    :return:
#    """

# Headers
# headers = [ "bankAccountId", "displayName", "amount", "plaidCategory", "transactionDate", "created"]
# Load the dataset into Pandas data frame
# dataset = read_data(INPUT_PATH)
# Add the headers to the loaded dataset
# dataset = add_headers(dataset, headers)
# Save the loaded dataset into csv format
# dataset.to_csv(OUTPUT_PATH, index=False)
# print "File saved ...!"


# def split_dataset(dataset, train_percentage, feature_headers, target_header):
#         """
#    Split the dataset with train_percentage
#    :param dataset:
#    :param train_percentage:
#    :param feature_headers:
#    :param target_header:
#    :return: train_x, test_x, train_y, test_y
#    """

# Split dataset into train and test dataset
# train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
#                                                     train_size=train_percentage)
# return train_x, test_x, train_y, test_y


# def handel_missing_values(dataset, missing_values_header, missing_label):
#         """
#    Filter missing values from the dataset
#    :param dataset:
#    :param missing_values_header:
#    :param missing_label:
#    :return:
#    """

# return dataset[dataset[missing_values_header] != missing_label]


# def random_forest_classifier(features, target):
#        """
#    To train the random forest classifier with features and target data
#    :param features:
#    :param target:
#    :return: trained random forest classifier
#    """
# clf = RandomForestClassifier()
# clf.fit(features, target)
# return clf


# def dataset_statistics(dataset):
#        """
#    Basic statistics of the dataset
#    :param dataset: Pandas dataframe
#    :return: None, print the basic statistics of the dataset
#    """
# print dataset.describe()


# def main():
#        """
#    Main function
#    :return:
#    """
# Load the csv file into pandas dataframe
# dataset = pd.read_csv(OUTPUT_PATH)
# Get basic statistics of the loaded dataset
# dataset_statistics(dataset)

# Filter missing values
# dataset = handel_missing_values(dataset, HEADERS[6], '?')
# train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

# Train and Test dataset size details
# print "Train_x Shape :: ", train_x.shape
# print "Train_y Shape :: ", train_y.shape
# print "Test_x Shape :: ", test_x.shape
# print "Test_y Shape :: ", test_y.shape

# Create random forest classifier instance
# trained_model = random_forest_classifier(train_x, train_y)
# print "Trained model :: ", trained_model
# predictions = trained_model.predict(test_x)

# for i in xrange(0, 5):
#        print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

# print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
# print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
# print " Confusion matrix ", confusion_matrix(test_y, predictions)


# if __name__ == "__main__":
#        main()