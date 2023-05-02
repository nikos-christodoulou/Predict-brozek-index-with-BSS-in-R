body = read.csv("C:/Users/Nikos/Dropbox/stat_ml/exercise_2/body.csv")
View(body)
attach(body)
library(car)
library(rattle)
library(MASS)
library(class)

sum(is.na(body))


#Before starting to find the optimal subset
#we take a look at the distribution of our data
#and the correlation that they have between them. 

library(PerformanceAnalytics)
chart.Correlation(body)

# From the correlation graph we can see that's there is
# a stronger correlation between brozek-density-adipos-chest-abdom.


# In order to find the optimal subset we will use three methods
# Best subset, forward-stepwise and backward-stepwise selection.

library(leaps)

#a i) Best Subset
regfit.full <- regsubsets(brozek~., data = body, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
names(reg.summary)
reg.summary$rsq

par(mfrow=c(2,2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
which.max(reg.summary$adjr2)
points(6, reg.summary$adjr2[6], col="red", cex=2, pch=20)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summary$cp)
plot(reg.summary$bic, xlab = "Number of Variables", ylab= "BIC", type = "l")
which.min(reg.summary$bic)

coef(regfit.full, 6)

#a ii)
### Forward / Backward Stepwise Selection ###

regfit.fwd <- regsubsets(brozek~., data = body, nvmax = 15, method = "forward")
reg.summaryfwd <- summary(regfit.fwd)
regfit.bwd <- regsubsets(brozek~., data = body, nvmax = 15, method = "backward")
reg.summarybwd <- summary(regfit.bwd)
names(reg.summaryfwd)
names(reg.summarybwd)
reg.summaryfwd$rsq
reg.summarybwd$rsq
reg.summaryfwd$bic
reg.summarybwd$bic

coef(regfit.full, 6)
coef(regfit.fwd, 6)
coef(regfit.bwd, 6)

#coefficients obtained from the full model and backward selection are the same.
#the coefficients obtained from forward selection are different.

coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)

# We notice that for the 7 variables model, we get similar coefficient values
# for the full model. This suggests that the selected variables for the 
# regression models are a good subset of predictors


par(mfrow=c(2,2))
plot(reg.summaryfwd$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summaryfwd$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
which.max(reg.summaryfwd$adjr2)
points(7, reg.summaryfwd$adjr2[7], col="red", cex=2, pch=20)
plot(reg.summaryfwd$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summaryfwd$cp)
points(10, reg.summaryfwd$cp[10],col="red", cex=2, pch = 20)
plot(reg.summaryfwd$bic, xlab = "Number of Variables", ylab= "BIC", type = "l")
which.min(reg.summaryfwd$bic)
points(6, reg.summaryfwd$bic[6], col = "red", cex = 2, pch = 20)


par(mfrow=c(2,2))
plot(reg.summarybwd$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summarybwd$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
which.max(reg.summarybwd$adjr2)
points(6, reg.summarybwd$adjr2[6], col="red", cex=2, pch=20)
plot(reg.summarybwd$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summarybwd$cp)
points(6, reg.summarybwd$cp[6],col="red", cex=2, pch = 20)
plot(reg.summarybwd$bic, xlab = "Number of Variables", ylab= "BIC", type = "l")
which.min(reg.summarybwd$bic)


#Now that we have a first picture for our subset
#we will apply the validation set approach and cross validation to see how each of these subsets
#behave while fitting.

set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(body), rep = TRUE)
test <- (!train)

# We apply regsubsets() to the training set in order to perform best
# subset selection:

regfit.best <- regsubsets(brozek~., data = body[train,], nvmax = 15)
reg.summarybest <- summary(regfit.best)
which.max(reg.summarybest$adjr2)
which.min(reg.summarybest$cp)
which.min(reg.summarybest$bic)



test.mat <- model.matrix(brozek~., data = body[test, ])

val.errors <- rep(NA, 15)
for (i in 1 : 15){
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[, names(coefi)]%*%coefi
  val.errors[i] <- mean((body$brozek[test]-pred)^2)
}

val.errors
which.min(val.errors)
coef(regfit.best, 6)

predict.regsubsets <- function(object, newdata, id){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars]%*%coefi
}



k = 10
set.seed(1)
folds <-sample(1:k, nrow(body), replace = TRUE)
cv.errors <- matrix(NA, k, 15, dimnames=list(NULL, paste(1:15)))

# Now we write a for loop that performs cross-validation. In the jth fold, the
# elements of folds that equal j are in the test set, and the remainder are in
# the training set. We make our predictions for each model size (using our
# new predict.regsubsets() method), compute the test errors on the appropriate 
# subset, and store them in the appropriate slot in the matrix cv.errors.

for (j in 1:k){
  best.fit <- regsubsets(brozek~., data = body[folds!=j,], nvmax=15)
  for (i in 1:15){
    pred <- predict.regsubsets(best.fit, body[folds==j,], id = i)
    cv.errors[j, i] <- mean((body$brozek[folds==j]-pred)^2)
  }
}

#We use the apply() function to average over the columns of this
# matrix in order to obtain a vector for which the jth element is 
# the cross-validation error for the j-variable model.

mean.cv.errors  <- apply(cv.errors,2, mean)
mean.cv.errors
plot(mean.cv.errors, type = "b")

reg.best <- regsubsets(brozek~., data = body, nvmax = 15)
coef(reg.best, 6)
#our optimal subset contains: density, age, weight, chest, ankle, biceps


#c
#Logistic Regression
#we need to create a binomial variable to predict if brozek value > 24% or not.
brozek_log <- (brozek > 24.0)
table(brozek_log)
bodyNew <- data.frame(body, brozek_log)
attach(bodyNew)
View(bodyNew)


#we create a train and test set to apply our models.
set.seed(1)
randData = sample(2, nrow(body), replace = TRUE, prob = c(0.7, 0.3))
train = bodyNew[randData == 1,]
test = bodyNew[randData == 2,]
print(train)
print(test)

glm.fit0 <- glm(brozek_log ~ density + age + weight + chest + ankle + biceps, data = train, family="binomial")
summary(glm.fit0)

probs.fit0 <- predict(glm.fit0, newdata = test, type = "response")
probs.train <- predict(glm.fit0, newdata = train, type = "response")

pred.trainlog <- ifelse(probs.train > 0.5, 1, 0)
pred.testlog <- ifelse(probs.fit0 > 0.5, 1, 0)

#evaluating logistic regression performace

mean(pred.testlog==test$brozek_log)

conf.mat <- table(pred.testlog, test$brozek_log)
conf.mat

err <-(conf.mat[1,2]+conf.mat[2,1])/sum(conf.mat)
err

# Fraction of correct predictions:
corpred <- 1-err
corpred
# True positive rate = Sensitivity = 1-Type II error:
tpr <- conf.mat[2,2]/sum(conf.mat[,2])
tpr
# False positive rate = 1 - Specificity = Type I error:
fpr <- conf.mat[2,1]/sum(conf.mat[,1])
fpr
# Specificity
spc <- 1-fpr
spc
# Positive predicted value = Precision
ppv <- conf.mat[2,2]/sum(conf.mat[2,])
ppv
# Negative predicted value
npv <- conf.mat[1,1]/sum(conf.mat[1,])
npv

#--- LDA ---

lda.fit <- lda(brozek_log ~ density + age + weight + chest + ankle + biceps, data = train)
lda.preds <- predict(lda.fit, newdata = test)
conf.mat <- table(lda.preds$class, test$brozek_log)

mean(lda.preds$class==test$brozek_log)

err <-(conf.mat[1,2]+conf.mat[2,1])/sum(conf.mat)
err

# Fraction of correct predictions:
corpred <- 1-err
corpred
# True positive rate = Sensitivity = 1-Type II error:
tpr <- conf.mat[2,2]/sum(conf.mat[,2])
tpr
# False positive rate = 1 - Specificity = Type I error:
fpr <- conf.mat[2,1]/sum(conf.mat[,1])
fpr
# Specificity
spc <- 1-fpr
spc
# Positive predicted value = Precision
ppv <- conf.mat[2,2]/sum(conf.mat[2,])
ppv
# Negative predicted value
npv <- conf.mat[1,1]/sum(conf.mat[1,])
npv



#we can see that lda > logistic regression

#--- QDA ---

qda.fit <- qda(brozek_log ~ density + age + weight + chest + ankle + biceps, data = train)
qda.preds <- predict(qda.fit, newdata = test)


conf.mat <- table(qda.preds$class, test$brozek_log)
conf.mat

mean(qda.preds$class==test$brozek_log)


err <-(conf.mat[1,2]+conf.mat[2,1])/sum(conf.mat)
err

# Fraction of correct predictions:
corpred <- 1-err
corpred
# True positive rate = Sensitivity = 1-Type II error:
tpr <- conf.mat[2,2]/sum(conf.mat[,2])
tpr
# False positive rate = 1 - Specificity = Type I error:
fpr <- conf.mat[2,1]/sum(conf.mat[,1])
fpr
# Specificity
spc <- 1-fpr
spc
# Positive predicted value = Precision
ppv <- conf.mat[2,2]/sum(conf.mat[2,])
ppv
# Negative predicted value
npv <- conf.mat[1,1]/sum(conf.mat[1,])
npv


#we see that we get a lower accuracy than the other two methods

#--- KNN ---

knn_train <- 1 : NROW(train)
data <- rbind(train, test)
View(data)
training_data <- data[knn_train,c("density", "age", "weight", "chest", "ankle", "biceps")]
testing_data <- data[-knn_train, c("density", "age", "weight", "chest", "ankle", "biceps")]
train.brozek <- bodyNew$brozek_log[knn_train]
test.brozek <- bodyNew$brozek_log[-knn_train]




kval <- seq(5, 50, by = 2)
knn.res <- matrix(ncol = 3, nrow = length(kval))
knn.res[,1] <- kval
for(i in 1: length(kval)){
  knn.pred <- knn(training_data, testing_data, train.brozek, k=kval[i])
  knn.res[i, 2] <- mean(knn.pred != test.brozek)
  knn.pred0 <- knn(training_data, training_data, train.brozek, k=kval[i])
  knn.res[i, 3] <- mean(knn.pred0 != train.brozek)
}
pp <- which.min(knn.res[,2])
pp
knn.res[pp,]


plot(knn.res[,1], knn.res[,2], type="l", xlab = "K", ylab="error rate", ylim=range(knn.res[,2:3]))
lines(knn.res[,1], knn.res[,3], col="blue")
abline(v=knn.res[pp,1],col="red",lty=2)
legend(40, 0.6, col=c("blue","black"),lty=rep(1,2), c("train","test"),bty="n")

#we see that the best result is for k = 9.
knn.pred <- knn(training_data, testing_data, train.brozek, k=9)
table(knn.pred, test.brozek)

table(knn.pred0, train.brozek)
table(knn.pred, test.brozek)


conf.mat <- table(knn.pred, test.brozek)
conf.mat
mean(knn.pred == test.brozek)

err <-(conf.mat[1,2]+conf.mat[2,1])/sum(conf.mat)
err

# Fraction of correct predictions:
corpred <- 1-err
corpred
# True positive rate = Sensitivity = 1-Type II error:
tpr <- conf.mat[2,2]/sum(conf.mat[,2])
tpr
# False positive rate = 1 - Specificity = Type I error:
fpr <- conf.mat[2,1]/sum(conf.mat[,1])
fpr
conf.mat[2,1]
sum(conf.mat[,1])
# Specificity
spc <- 1-fpr
spc
# Positive predicted value = Precision
ppv <- conf.mat[2,2]/sum(conf.mat[2,])
ppv
# Negative predicted value
npv <- conf.mat[1,1]/sum(conf.mat[1,])
npv

