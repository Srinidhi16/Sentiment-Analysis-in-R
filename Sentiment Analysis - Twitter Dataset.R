#Sentiment Analysis - Twitter Dataset.R

#The goal of this project is to see if our machine learning model can accurately, to a large extent, predict if a sentiment will be positive or negative by just going through the tweets.
#install.packages("magrittr") # package installations are only needed the first time you use it
#install.packages("dplyr")
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)
#Import Data
getwd()
setwd("C:\\projects\\New folder")
df = read.csv("Sentiment.csv")
head(df)

#Get the text and the sentiment columns.
#install.packages("tidyverse",dependencies = TRUE)

library(tidyverse)
df1 <- df %>% 
  select(text, sentiment)
head(df1)

#Structure of dataset
str(df1)

#Number of positive, negative and neutral texts
table(df1$sentiment)

#Get the proportion 
round(prop.table(table(df1$sentiment)),2)

#Here we can see  proportion of sentiments in our dataset is that we have 61% as negative, 23% as neutral and 16% as positive.
           
#DATA CLEANING
#install.packages("tm")
#install.packages("SnowballC")
#install.packages("corpus")
#install.packages("VCorpus")
library(tm)
library(SnowballC)
library(corpus)

#A vector source interprets each element of the vector x as a document.
#VCorpus() takes a source object and makes a volatile corpora.
corpus = Corpus(VectorSource(df1$text))
corpus

#Sample of a text stored in corpus
as.character(corpus[[5]])

#install.packages("textstem")
library(textstem)

#Convert text to lower case, remove numbers, remove puntuation, stopwords and so on.
corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, stemDocument, language = "english")
corpus = tm_map(corpus, stripWhitespace)
as.character(corpus[[1]])

#Creating the Document Term Matrix for the model
#mathematical matrix that describes the frequency of terms that occur in a collection of documents.

dtm <- DocumentTermMatrix(corpus)
dtm
dim(dtm)

#The document-term matrix presently has 16272 words extracted from 13,871 tweets. These words are what we will use to decide if a tweet is positive or negative.
#The sparcity of the dtm is 100% which means no words is left out the matrix.

dtm = removeSparseTerms(dtm, 0.999)
dim(dtm)

#Inspecting the the first 10 tweets and the first 10 words in the dataset
inspect(dtm[0:10, 1:10])

freq<- sort(colSums(as.matrix(dtm)), decreasing=TRUE)

#Words that appeared more than 60 times.
findFreqTerms(dtm, lowfreq=60) #identifying terms that appears more than 60times

library(ggplot2)
wf<- data.frame(word=names(freq), freq=freq)
head(wf)

library("wordcloud")
#Positive cloud
positive <- subset(df1,sentiment=="Positive")
head(positive)
wordcloud(positive$text, max.words = 500, scale = c(3,0.5))

#Negative cloud
negative <- subset(df1,sentiment=="Negative")
head(negative)
wordcloud(negative$text, max.words = 500, scale = c(3,0.5))

#Neutral cloud
neutral <- subset(df1,sentiment=="Neutral")
head(neutral)
wordcloud(neutral$text, max.words = 500, scale = c(3,0.5))

## Loading required package: RColorBrewer
#install.packages("RColorBrewer")
library(RColorBrewer)
library("wordcloud")

set.seed(1234)
wordcloud(words = wf$word, freq = wf$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

## Apply the convert_count function to get final training and testing DTMs
d <- apply(dtm, 2, convert_count)
#head(d)

dataset = as.data.frame(as.matrix(d))
head(dataset)

dataset$Class = df1$sentiment
str(dataset$Class)

head(dataset)
dim(dataset)

#Data Splitting 
library(caret)

intrain <- createDataPartition(y = dataset$Class, p= 0.7, list = FALSE)
training <- dataset[intrain,]
testing <- dataset[-intrain,]
dim(training)
dim(testing)

#We need to factorize them:               
training[["Class"]] = factor(training[["Class"]])
testing[["Class"]] = factor(testing[["Class"]])

sum(is.na(training))
sum(is.na(testing))

#MODEL TRAINING
#Model 1: NAIVE BAYES
#install.packages('e1071')

library(e1071)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time( classifier_nb <- naiveBayes(training, training$Class, laplace = 1,
                                         trControl = control,tuneLength = 7) )
nb_pred = predict(classifier_nb, type = 'class', newdata = testing)

#Check the type of variable
typeof(testing$Class)
typeof(nb_pred)

confusionMatrix(data = nb_pred,reference = testing$Class)


#===========================================================================
#Model 2: SVM
#install.packages("kernlab")

#[This will control all the computational overheads so that we can use the train() function provided by the caret package. The training method will train our data on different algorithms.
#trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#The "method" parameter defines the resampling method, in this demo we'll be using the repeatedcv or the repeated cross-validation method.
#The next parameter is the "number", this basically holds the number of resampling iterations.
#The "repeats " parameter contains the sets to compute for our repeated cross-validation. We are using setting number =10 and repeats =3]

library(dplyr)
library(kernlab)


svm_classifier <- ksvm(Class ~ ., data = training,kernel = "polydot", na.action = na.omit)
svm_classifier

svm_pred = predict(svm_classifier,testing)

confusionMatrix(svm_pred,testing$Class)

#===============================================================================================
#Model 3: RANDOM FOREST
#install.packages("randomForest")

library(randomForest)
library(caTools)

typeof(training$Class)
#testing$Class <- as.factor(testing$Class)

rf_classifier = randomForest(x = training,
                             y = as.factor(training$Class),
                             ntree = 25)

rf_classifier

# Predicting the Test set results
#levels(train$Class) <- c(levels(train$Class),"3448")
#testing$Class <- factor(testing$Class, levels = levels(training$Class))

rf_pred = predict(rf_classifier, newdata = testing)

# Making the Confusion Matrix
library(caret)

confusionMatrix(table(rf_pred,testing$Class))

#Conclusion
#On comparing the scores using confusion matrix for each of the 3 models, Naive Bayes model performs the best with 98% accuracy and Random Forest model with 97% as compared to Support Vector Machine  67%. Naive Bayes works on the assumption that the features of the dataset are independent of each other hence called Naive.
#It works well for bag-of-words models a.k.a text documents since words in a text document are independent of each other; the location of one word doesn't depend on another word.Hence, it satisfys the independence assumption of the Naive Bayes model.
#It is therefore the most commonly used model for text classification, sentiment analysis, spam filtering & recommendation systems.
