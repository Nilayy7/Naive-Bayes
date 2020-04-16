#Prepare a classification model using Naive Bayes for salary data .
# Libraries
install.packages("naivebayes")
library(naivebayes)
library(ggplot2)
library(caret)

#Training Data
trainsal <- SalaryData_Train
str(trainsal)
trainsal$educationno <- as.factor(trainsal$educationno)
class(trainsal)

#Testing Data
testsal <- SalaryData_Test
str(testsal)
testsal$educationno <- as.factor(testsal$educationno)
class(testsal)

#Let us do some EDA
#Boxplot
ggplot(data = trainsal,aes(x=trainsal$Salary,y=trainsal$age,fill=trainsal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")


ggplot(data = trainsal,aes(x=trainsal$Salary,y=trainsal$capitalgain,fill=trainsal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")

plot(trainsal$workclass,trainsal$Salary)
plot(trainsal$maritalstatus,trainsal$Salary)
plot(trainsal$relationship,trainsal$Salary)
plot(trainsal$sex,trainsal$Salary)

ggplot(data = trainsal,aes(x=trainsal$Salary,y=trainsal$hoursperweek,fill=trainsal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")

#Density Plot
ggplot(data = trainsal,aes(x=trainsal$age,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data = trainsal,aes(x=trainsal$workclass,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data = trainsal,aes(x=trainsal$sex,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data = trainsal,aes(x=trainsal$relationship,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data = trainsal,aes(x=trainsal$native,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data = trainsal,aes(x=trainsal$race,fill=trainsal$Salary))+
  geom_density(alpha = 0.9, color = 'Violet')

#Naive Bayes Model

model <- naive_bayes(trainsal$Salary~.,data = trainsal)
model

#Predictions On Test Data
pred <- predict(model,testsal)
mean(pred == testsal$Salary) #81.87% is the Accuracy

#Confusion Matrix
confusionMatrix(pred,testsal$Salary)
