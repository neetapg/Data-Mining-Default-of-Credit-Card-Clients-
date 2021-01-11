
library(forecast)
library(caret)
library(ggplot2)
library(gains)
library(rpart)
library(rpart.plot)
library(adabag)
library(randomForest)
library(neuralnet)
library(scales)
library(dummies)


# Reading the data
defaulter.df <- read.csv(file="Default.csv",header=TRUE)
summary(defaulter.df)
dim(defaulter.df)
colnames(defaulter1.df)


# dropping the ID Column
defaulter1.df<-defaulter.df[,-1]


# changing the categorical variables to Factor
defaulter1.df$SEX<-as.factor(defaulter1.df$SEX)
defaulter1.df$EDUCATION<-as.factor(defaulter1.df$EDUCATION)
defaulter1.df$MARRIAGE<-as.factor(defaulter1.df$MARRIAGE)
defaulter1.df$PAY_Sep<-as.factor(defaulter1.df$PAY_Sep)
defaulter1.df$PAY_Aug<-as.factor(defaulter1.df$PAY_Aug)
defaulter1.df$PAY_Jul<-as.factor(defaulter1.df$PAY_Jul)
defaulter1.df$PAY_Jun<-as.factor(defaulter1.df$PAY_Jun)
defaulter1.df$PAY_May<-as.factor(defaulter1.df$PAY_May)
defaulter1.df$PAY_Apr<-as.factor(defaulter1.df$PAY_Apr)

lapply(defaulter1.df, class) # to check if they are shown as factor

############################################################################################
###################################### Visualization #######################################

# boxplot for all the columns of dataset
boxplot(defaulter.df)


# visualization of payment status per month (one separate plot for each months)
par(mfcol = c(1,1))
plot(defaulter.df$PAY_Sep,xlab = "repayment status", ylab = "Frequency", main="Status in september 2005")
plot(defaulter.df$PAY_Aug,xlab = "repayment status", ylab = "Frequency",main="Status in August 2005")
plot(defaulter.df$PAY_Jul,xlab = "repayment status", ylab = "Frequency", main="Status in July 2005")
plot(defaulter.df$PAY_Jun,xlab = "repayment status", ylab = "Frequency", main="Status in June 2005")
plot(defaulter.df$PAY_May,xlab = "repayment status", ylab = "Frequency",main="Status in May 2005")
plot(defaulter.df$PAY_Apr,xlab = "repayment status", ylab = "Frequency",main="Status in April 2005")


# creating different copy of Dataset for manipulation required for visualization
defaulter2.df <- defaulter1.df
defaulter2.df$SEX = ifelse(defaulter2.df$SEX == 1, "Male", "Female")
defaulter2.df$default.payment.next.month <- as.factor(defaulter2.df$default.payment.next.month)



# Bar Graph for defaulters based on the gender
ggplot(data = defaulter2.df, mapping = aes(x = SEX, fill = default.payment.next.month)) +
  geom_bar() +
  ggtitle("Gender") +
  stat_count(aes(label = ..count..), geom = "label")


# graph for defaulters based on the education
ggplot(data = defaulter2.df, mapping = aes(x = EDUCATION, fill = default.payment.next.month)) +
  geom_bar() +
  ggtitle("EDUCATION") +
  stat_count(aes(label = ..count..), geom = "label")


# graph for defaulters based on the marital status
ggplot(data = defaulter2.df, mapping = aes(x = MARRIAGE, fill = default.payment.next.month)) +
  geom_bar() +
  ggtitle("MARRIAGE") +
  stat_count(aes(label = ..count..), geom = "label")



# LIMIT_BAL density plot
ggplot(data = defaulter2.df, mapping = aes(x = LIMIT_BAL)) + 
  geom_density(fill = "#000080") +
  ggtitle("LIMIT_BAL Distribution") +
  xlab("LIMIT_BAL") +
  geom_vline(xintercept = mean(defaulter2.df$LIMIT_BAL), col = "red", 
             linetype = "dashed", size = 0.6) +
  annotate("text", 
           x = -Inf, y = Inf, 
           label = paste("Mean:", round(mean(defaulter2.df$LIMIT_BAL), digits = 2)), 
           hjust = 0, vjust = 1, col = "red", size = 3)


################################################################################
# partitioning the data to Training(%60) and Validation(%40) sections


set.seed(1) 

train.rows <- sample(rownames(defaulter1.df), dim(defaulter1.df)[1]*0.6)
train.data <- defaulter1.df[train.rows, ]

valid.rows <- setdiff(rownames(defaulter1.df), train.rows) 
valid.data <- defaulter1.df[valid.rows, ]

dim(defaulter.df)
dim(train.data)
dim(valid.data)


# making some adjustments in the splits, making sure some rare cases will exist in the training data
summary(train.data$PAY_Jun)
summary(valid.data$PAY_Jun)
summary(train.data$PAY_May)
summary(valid.data$PAY_May)

Jun_level1_row<-valid.data[valid.data$PAY_Jun==1,]
Jun_level1_row
valid.data<-valid.data[-c(6783,11498),]
train.data<-rbind(train.data,Jun_level1_row)
dim(valid.data)
dim(train.data)

May_level8_row<-valid.data[valid.data$PAY_May==8,]
May_level8_row
valid.data<-valid.data[-c(8655),]
train.data<-rbind(train.data,May_level8_row)
dim(valid.data)
dim(train.data)


################################################################################
############################# Logit Model ######################################

# model1 (including all the variables)
banklogit1.reg <- glm(default.payment.next.month ~ ., data = train.data, family = "binomial") 

# model2 (without pay_0 to 6)
banklogit2.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+ 
                       BILL_AMT_Sep + BILL_AMT_Aug + BILL_AMT_Jul + BILL_AMT_Jun + BILL_AMT_May + BILL_AMT_Apr 
                     + PAY_AMT_Sep + PAY_AMT_Aug + PAY_AMT_Jul + PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, 
                     data = train.data, family = "binomial") 


# model3 (including the months up to August)
summary(train.data$PAY_Jul)

banklogit3.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+
                       +PAY_Aug+  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                     +BILL_AMT_Aug +BILL_AMT_Jul+ BILL_AMT_Jun + BILL_AMT_May+BILL_AMT_Apr
                     +PAY_AMT_Aug +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, 
                     data = train.data, family = "binomial") 

# model4 (including the months up to July)
summary(train.data$PAY_Jul)
banklogit4.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+
                       PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                     +BILL_AMT_Jul+ BILL_AMT_Jun + BILL_AMT_May+BILL_AMT_Apr
                     +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, 
                     data = train.data, family = "binomial") 


# model5 (including the months up to June)
banklogit5.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+
                       +PAY_Jun+PAY_May+PAY_Apr
                     + BILL_AMT_Jun + BILL_AMT_May+BILL_AMT_Apr
                     +PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, 
                     data = train.data, family = "binomial") 
# model6 (including the months up to May)
banklogit6.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+
                       +PAY_May+PAY_Apr
                     + BILL_AMT_May+BILL_AMT_Apr
                     + PAY_AMT_May + PAY_AMT_Apr, 
                     data = train.data, family = "binomial") 
# model7 (including Just the month Apr)
banklogit7.reg <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE+
                       +PAY_Apr
                     +BILL_AMT_Apr
                     + PAY_AMT_Apr, 
                     data = train.data, family = "binomial")


options(scipen=999)
summary(banklogit.reg)
hist(banklogit7.reg$residuals)

summary(valid.data)
  # if sex2(beta=-0.138) female(***)        ---> Odds of belonging to class=1(defaulter) will decrease
  # if limit_bal(***), education4(*),5(***) ---> Odds of belonging to class=1(defaulter) will decrease

banklogit.reg.pred <- predict(banklogit1.reg, valid.data[, -24], type = "response") 
df<-data.frame(actual.defaulter = valid.data$default.payment.next.month, predicted.defaulter = banklogit.reg.pred)
#View(df)


# generating the confusion matrix
confusionMatrix(as.factor(ifelse(banklogit.reg.pred>0.5,1,0)), as.factor(valid.data[, 24]),positive = '1' )


# Generating the data for Lift chart & decile-wise chart
bankgain <- gains(valid.data$default.payment.next.month, banklogit.reg.pred, groups=10)
bankgain


#Lift Index is Mean Resp / Cum MEan Resp for the last row*100
# plot lift chart
plot(c(0,bankgain$cume.pct.of.total*sum(valid.data$default.payment.next.month))~c(0,bankgain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(valid.data$default.payment.next.month))~c(0, dim(valid.data)[1]), lty=2)


# compute deciles and plot decile-wise chart
heights <- bankgain$mean.resp/mean(valid.data$default.payment.next.month)
heights
decileplot <- barplot(heights, names.arg = bankgain$depth, ylim = c(0,9), 
                      xlab = "Percentile", ylab = "Mean Response/Overall Mean", main = "Decile-wise lift chart")
text(decileplot, heights+0.5, labels=round(heights, 1), cex = 0.8) # add labels to columns



################################################################################
############################# CART Model ######################################

# model1 (Including all the 6 months)
defaultcv1.ct <- rpart(default.payment.next.month ~ ., data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)

# model2(till august)
defaultcv2.ct <- rpart(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                        MARRIAGE + AGE+
                        +PAY_Aug+  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                      +BILL_AMT_Aug +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                      +PAY_AMT_Aug +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)

# model3(till July)
defaultcv3.ct <- rpart(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                        MARRIAGE + AGE+
                        +  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                      +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                      +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)

# model4(till June)
defaultcv4.ct <- rpart(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                        MARRIAGE + AGE+
                        +PAY_Jun+PAY_May+PAY_Apr
                      + BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                      +PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)

# model5(till May)
defaultcv5.ct <- rpart(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                        MARRIAGE + AGE+
                        +PAY_May+PAY_Apr
                      +BILL_AMT_May+BILL_AMT_Apr
                      + PAY_AMT_May + PAY_AMT_Apr, data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)

# model6(till Apr)
defaultcv6.ct <- rpart(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                        MARRIAGE + AGE+
                        +PAY_Apr
                      +BILL_AMT_Apr
                      + PAY_AMT_Apr, data = train.data, method = "class",cp = 0.001, minsplit = 2,xval=5)


# count number of leaves
length(defaultcv1.ct$frame$var[defaultcv1.ct$frame$var == "<leaf>"])

# plot tree
prp(defaultcv1.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, cex=0.5,
    box.col=ifelse(defaultcv1.ct$frame$var == "<leaf>", 'gray', 'white'))  

# print the CP Table
printcp(defaultcv1.ct)



# model1
defaultpruned1.ct <- prune(defaultcv1.ct, 
                          cp = 0.002)
# model2
defaultpruned2.ct <- prune(defaultcv2.ct, 
                          cp = 0.002)
# model3
defaultpruned3.ct <- prune(defaultcv3.ct, 
                          cp = 0.003)
# model4
defaultpruned4.ct <- prune(defaultcv4.ct, 
                          cp = 0.03)
# model5
defaultpruned5.ct <- prune(defaultcv5.ct, 
                          cp = 0.002)
# model6
defaultpruned6.ct <- prune(defaultcv6.ct, 
                          cp = 0.006)

prp(defaultpruned1.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(defaultpruned1.ct$frame$var == "<leaf>", 'gray', 'white'))  

defaultprune.pred.valid <- predict(defaultpruned1.ct,valid.data,type = "class")
confusionMatrix(defaultprune.pred.valid, as.factor(valid.data$default.payment.next.month), positive ='1' )



#############################
# Boosted Tree models


# model1
defaultboost1 <- boosting(default.payment.next.month ~ ., data = train.data)

# model2
defaultboost2 <- boosting(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                           MARRIAGE + AGE+
                           +PAY_Aug+  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                         +BILL_AMT_Aug +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                         +PAY_AMT_Aug +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data)
# model3
defaultboost3 <- boosting(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                           MARRIAGE + AGE+
                           +  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                         +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                         +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data)

# model4
defaultboost4 <- boosting(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                           MARRIAGE + AGE+
                           +PAY_Jun+PAY_May+PAY_Apr
                         + BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                         +PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data)

# model5
defaultboost5 <- boosting(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                           MARRIAGE + AGE+
                           +PAY_May+PAY_Apr
                         +BILL_AMT_May+BILL_AMT_Apr
                         + PAY_AMT_May + PAY_AMT_Apr, data = train.data)

# model6
defaultboost6 <- boosting(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION +
                           MARRIAGE + AGE+
                           +PAY_Apr
                         +BILL_AMT_Apr
                         + PAY_AMT_Apr, data = train.data)

# Predict using Valid data
defaultBoost.pred.valid <- predict(defaultboost1,valid.data,type = "class")
# generate confusion matrix for validation data
confusionMatrix(as.factor(defaultBoost.pred.valid$class), as.factor(valid.data$default.payment.next.month ), positive ='1')



        ##### Error in the bosted

#############################
# Random Forest models

# Model1
defaultRF1 <- randomForest(as.factor(default.payment.next.month) ~ ., data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE)  

# Model2
defaultRF2 <- randomForest(as.factor(default.payment.next.month) ~ LIMIT_BAL + SEX + EDUCATION +
                            MARRIAGE + AGE+
                            +PAY_Aug+  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                          +BILL_AMT_Aug +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                          +PAY_AMT_Aug +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE) 
# Model3
defaultRF3 <- randomForest(as.factor(default.payment.next.month) ~ LIMIT_BAL + SEX + EDUCATION +
                            MARRIAGE + AGE+
                            +  PAY_Jul+PAY_Jun+PAY_May+PAY_Apr
                          +BILL_AMT_Jul+ BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                          +PAY_AMT_Jul+PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE)
# Model4
defaultRF4 <- randomForest(as.factor(default.payment.next.month) ~ LIMIT_BAL + SEX + EDUCATION +
                            MARRIAGE + AGE+
                            +PAY_Jun+PAY_May+PAY_Apr
                          + BILL_AMT_Jun +BILL_AMT_May+BILL_AMT_Apr
                          +PAY_AMT_Jun + PAY_AMT_May + PAY_AMT_Apr, data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE) 
# Model5
defaultRF5 <- randomForest(as.factor(default.payment.next.month) ~ LIMIT_BAL + SEX + EDUCATION +
                            MARRIAGE + AGE+
                            +PAY_May+PAY_Apr
                          +BILL_AMT_May+BILL_AMT_Apr
                          + PAY_AMT_May + PAY_AMT_Apr, data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE) 
# Model6
defaultRF6 <- randomForest(as.factor(default.payment.next.month) ~ LIMIT_BAL + SEX + EDUCATION +
                            MARRIAGE + AGE+
                            +PAY_Apr
                          +BILL_AMT_Apr
                          + PAY_AMT_Apr, data = train.data, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE) 

defaultRF.pred.valid <- predict(defaultRF1,valid.data,type = "class")
confusionMatrix(defaultRF.pred.valid, as.factor(valid.data$default.payment.next.month), positive ='1')

# variable importance plot
varImpPlot(defaultRF1, type = 1)



################################################################################
############################### NN Model #######################################

# Doing preProcessing needed for NN 

# dropping the ID Column
defaulter3.df<-defaulter.df[,-1]


# changing the categorical (nominal) variables to Factor
defaulter3.df$SEX<-as.factor(defaulter3.df$SEX)
defaulter3.df$EDUCATION<-as.factor(defaulter3.df$EDUCATION)
defaulter3.df$MARRIAGE<-as.factor(defaulter3.df$MARRIAGE)
defaulter3.df$default.payment.next.month <- as.factor(defaulter3.df$default.payment.next.month)

lapply(defaulter3.df, class) # to check if latest status of variables types

# Create dummies for Categorical (nominal) Variabes
default_dummy <- model.matrix(~ 0+SEX
                              +EDUCATION
                              +MARRIAGE,
                              data = defaulter3.df)
default_dummy.df <- as.data.frame(default_dummy) # Converting to a df
default_dummy.df <- default_dummy.df[,-c(2,8,11)]
head(default_dummy.df)
defaulter3.df <- cbind(defaulter3.df,default_dummy.df) # adding to the main df
dim(defaulter3.df)
summary(defaulter3.df)


# Scaling the Numeric variables (including the ordinal categorical variables) 
vars.to.scale <- c(1,5,6:11,12,13,14,15,16,17,18,19,20,21,22,23)
defaulter3.df[,vars.to.scale] <-lapply(defaulter3.df[,vars.to.scale], rescale)
summary(defaulter3.df)


# Partitioning the data after doing the manipulation required

set.seed(1) 

train.rows <- sample(rownames(defaulter3.df), dim(defaulter3.df)[1]*0.6)
valid.rows <- setdiff(rownames(defaulter3.df), train.rows)

train.data <- defaulter3.df[train.rows, ]
valid.data <- defaulter3.df[valid.rows, ]

dim(train.data)
dim(valid.data)

summary(train.data)


# Trying to create different combinations of the models

# Model 1: No Pay_#
n <- names(train.data[,-c(2,3,4,6:11,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3, act.fct = "logistic")
plot(banknn, rep="best")

banknn.predict <- compute(banknn, valid.data[,-c(2,3,4,6:11,24)])

default.predict <- as.factor(ifelse(banknn.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))

# Model 2: Apr
n <- names(train.data[,-c(2,3,4,6:10,12:16,18:22,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn2 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3,act.fct = "logistic")
plot(banknn2, rep="best")

banknn2.predict <- compute(banknn2, valid.data[,-c(2,3,4,6:10,12:16,18:22,24)])

default2.predict <- as.factor(ifelse(banknn2.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default2.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 3: Apr-May
n <- names(train.data[,-c(2,3,4,6:9,12:15,18:21,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn3 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3,act.fct = "logistic")
plot(banknn3, rep="best")

banknn3.predict <- compute(banknn3, valid.data)

default3.predict <- as.factor(ifelse(banknn3.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default3.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 4: Apr-June
n <- names(train.data[,-c(2,3,4,6:8,12:14,18:20,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn4 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3,act.fct = "logistic")
plot(banknn4, rep="best")

banknn4.predict <- compute(banknn4, valid.data)

default4.predict <- as.factor(ifelse(banknn4.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default4.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 5: Apr-Jul
n <- names(train.data[,-c(2,3,4,6:7,12:13,18:19,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn5 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3, act.fct = "logistic")
plot(banknn5, rep="best")

banknn5.predict <- compute(banknn5, valid.data)

default5.predict <- as.factor(ifelse(banknn5.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default5.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 6: Apr-Aug
n <- names(train.data[,-c(2,3,4,6,12,18,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn6 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3,act.fct = "logistic")
plot(banknn6, rep="best")

banknn6.predict <- compute(banknn6, valid.data)

default6.predict <- as.factor(ifelse(banknn6.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default6.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 7: Apr-Sep ALL
n <- names(train.data[,-c(2,3,4,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn7 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = 3,act.fct = "logistic")
banknn7.predict <- compute(banknn7, valid.data[,-c(2,3,4,24)])

default7.predict <- as.factor(ifelse(banknn7.predict$net.result[,1]>0.8,0,1))
confusionMatrix(reference = as.factor(valid.data$default.payment.next.month),
                data = as.factor(default7.predict),
                positive = '1', 
                dnn = c('Prediction','Actual'))

# Model 8: Apr-Aug
n <- names(train.data[,-c(2,3,4,6,12,18,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
head(train.data[1:1000,])
banknn8 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = c(1),act.fct = "logistic")
plot(banknn6, rep="best")

banknn8.predict <- compute(banknn8, valid.data)

default8.predict <- as.factor(ifelse(banknn8.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default8.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 9: Apr-Aug
n <- names(train.data[,-c(2,3,4,6,12,18,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn9 <- neuralnet(f,data = train.data[1:1000,], linear.output = F, hidden = c(1,1),act.fct = "logistic")
plot(banknn9, rep="best")

banknn9.predict <- compute(banknn9, valid.data)

default9.predict <- as.factor(ifelse(banknn9.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default9.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


# Model 10: Apr-Aug
n <- names(train.data[,-c(2,3,4,6,12,18,24)])
f <- as.formula(paste("default.payment.next.month ~", 
                      paste(n[!n %in% c("default.payment.next.month")], 
                            collapse = " + ")))
banknn10 <- neuralnet(f,data = train.data, linear.output = F, hidden = c(1,1,1),act.fct = "logistic")
plot(banknn10, rep="best")

banknn10.predict <- compute(banknn10, valid.data)

default10.predict <- as.factor(ifelse(banknn10.predict$net.result[,1]>0.5,0,1))
confusionMatrix(data = as.factor(default10.predict),
                reference = as.factor(valid.data$default.payment.next.month),
                positive = '1', 
                dnn = c('Prediction','Actual'))


