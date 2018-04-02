install.packages("pROC")
install.packages("ranger")
install.packages("caTools")
install.packages("varImp")
install.packages("randomForest")
install.packages("magrittr")
library(dplyr)
library(magrittr)
library(corrplot)
library(prcomp)
library(kernlab)
library(pROC)
library(rpart)
library(MASS)
library(caret)
library(caTools)
library(ranger)
library(e1071)
library(varImp)
library(randomForest)
library(tidyverse)

# loading dataset
data <- read.csv("data.csv", header =TRUE)
head(data)
row<- nrow(data)
col <- ncol(data)
str(data)
print(paste("The number of columns is", row))
print(paste("The number of columns is", col))


#Cleaning data set (Removing column 33 since all values are "NA" & removing the id column)
cleaned_dataset <- data[,3:32]
head(cleaned_dataset)
str(cleaned_dataset)

data$diagnosis <- as.factor(data$diagnosis)
round(prop.table(table(data$diagnosis)), 2)

#Checking coorelation
corr_mat <- round(cor(cleaned_dataset, method = "pearson"),1)
df2 <- cleaned_dataset %>% 
  dplyr::select(-findCorrelation(corr_mat, cutoff = 0.9))
ncol(df2)
nrow(df2)
head(df2)
str(df2)
colnames(df2)
colnames(cleaned_dataset)

#Plotting the correlation


corrplot(corr_mat, tl.col = "black", method = "shade",type = "lower",mar=c(0,1,0,1), tl.srt = 40)

df3 <- cleaned_dataset[,c(4,5,6,9,10,11,12,15,16,17,18,19,20,22,25,26,27,28,29,30)]
colnames(df3)
corrplot(round(cor(df3),1), tl.col = "black", method = "shade", type = "lower",mar=c(0,1,0,1), tl.srt = 40)

# Training and Testing dataset

df4 <- cbind(diagnosis = data$diagnosis, df3)
head(df4)
str(df4)

set.seed(1234)
data_index <- createDataPartition(df4$diagnosis, p=0.7, list = FALSE)
train_data <- df4[data_index,]
test_data <- df4[-data_index,]


#PCA 
PCA_Graph <- prcomp(cleaned_dataset, center = TRUE, scale = TRUE)
plot(PCA_Graph, type="l", main = " ")
grid(nx = 10, ny = 14)
title(main = "Principal components Analysis", sub = NULL, xlab = "Components")
box()
summary(PCA_Graph)

#Applying machine learning models

fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

#SVM Radial

model_svm <- train(diagnosis~.,
                   train_data,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)
pred_svm <- predict(model_svm, test_data)
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
cm_svm

pred_prob_lda <- predict(model_svm, test_data, type="prob")
roc_lda <- roc(test_data$diagnosis, pred_prob_lda$M)
plot(roc_lda)

colAUC(pred_prob_lda, test_data$diagnosis, plotROC=TRUE)

#SVM Linear

model_svm2 <- train(diagnosis~.,
                   train_data,
                   method="svmLinear",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)
pred_svm2 <- predict(model_svm2, test_data)
cm_svm2 <- confusionMatrix(pred_svm2, test_data$diagnosis, positive = "M")
cm_svm2

#SVM poly

model_svm3 <- train(diagnosis~.,
                    train_data,
                    method="svmPoly",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    trControl=fitControl)
pred_svm3 <- predict(model_svm3, test_data)
cm_svm3 <- confusionMatrix(pred_svm3, test_data$diagnosis, positive = "M")
cm_svm3


#KNN 

model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)
pred_knn <- predict(model_knn, test_data)
cm_knn <- confusionMatrix(pred_knn, test_data$diagnosis, positive = "M")
cm_knn


pred_prob_knn <- predict(model_knn, test_data, type="prob")
roc_knn <- roc(test_data$diagnosis, pred_prob_knn$M)
plot(roc_knn)

colAUC(pred_prob_knn, test_data$diagnosis, plotROC=TRUE)

#Random forest

model_rf <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric="ROC",
                  tuneLength=10,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)


pred_rf <- predict(model_rf, test_data)
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis, positive = "M")
cm_rf

pred_prob_rf <- predict(model_rf, test_data, type="prob")
roc_rf <- roc(test_data$diagnosis, pred_prob_rf$M)
plot(roc_rf)

model_rf$bestTune
plot(model_rf)
colAUC(pred_prob_rf, test_data$diagnosis, plotROC=TRUE)

plot(varImp(model_rf), top = 5, main = "Random forest")


modelrf.rf <- randomForest(diagnosis~., data=test_data, ntree=500, keep.forest=FALSE,
                          importance=TRUE)
plot(modelrf.rf, log="y", main ="Random forest")
varImpPlot(modelrf.rf,main = "Features Importance")
head(data)
head(test_data)