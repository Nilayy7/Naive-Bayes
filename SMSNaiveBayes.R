#Build a naive Bayes model on the data set for classifying the ham and spam

#Import Libraries
ham_spam_SMS_messages <- sms_raw_NB
names(ham_spam_SMS_messages) <- c("Type", "Text")
str(ham_spam_SMS_messages)

ham_spam_SMS_messages$Type <- factor(ham_spam_SMS_messages$Type)
str(ham_spam_SMS_messages$Type)
table(ham_spam_SMS_messages$Type)

library(tm)
ham_spam_SMS_messages_corpus <- VCorpus(VectorSource(ham_spam_SMS_messages$Text))

ham_spam_SMS_messages_corpus
## <<VCorpus>>
inspect(ham_spam_SMS_messages_corpus[1:3])
as.character(ham_spam_SMS_messages_corpus[[1]])
lapply(ham_spam_SMS_messages_corpus[1:6], as.character)

#Make all characters lowercase,Remove all punctuation marks,Remove numbers
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_messages_corpus, content_transformer(tolower))
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_corpus_clean, removePunctuation)
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_corpus_clean, removeNumbers)

library(SnowballC)
#Remove Whitespace, Remove Words
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_corpus_clean, stripWhitespace)
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_corpus_clean, removeWords, stopwords())
ham_spam_SMS_corpus_clean <- tm_map(ham_spam_SMS_corpus_clean, stemDocument)
library(wordcloud)
wordcloud(ham_spam_SMS_corpus_clean, scale=c(2,.5), min.freq = 10, max.words = 300,
          random.order = FALSE, rot.per = .5, 
          colors= palette())

#Creation Of DTM Matrix
ham_spam_SMS_corpus_dtm <- DocumentTermMatrix(ham_spam_SMS_corpus_clean)
ham_spam_SMS_corpus_dtm

ham_spam_SMS_corpus_dtmMAtrix <- as.matrix(ham_spam_SMS_corpus_dtm)
ham_spam_SMS_corpus_dtmMAtrix[1:10, 1:10]

#Creating Training and Testing Data
ham_spam_SMS_training_set <- ham_spam_SMS_corpus_dtm[1:3901, ]
ham_spam_SMS_test_set <- ham_spam_SMS_corpus_dtm[3901:5559, ]

ham_spam_SMS_training_set_Labels <- ham_spam_SMS_messages[1:3901, ]$Type
ham_spam_SMS_test_set_Labels <- ham_spam_SMS_messages[3901:5559, ]$Type
prop.table(table(ham_spam_SMS_training_set_Labels))
prop.table(table(ham_spam_SMS_test_set_Labels))


#Reading Word Frequency
ham_spam_SMS_freq_words <- findFreqTerms(ham_spam_SMS_training_set, 10)
str(ham_spam_SMS_freq_words)

ham_spam_SMS_training_set_freq10 <- ham_spam_SMS_training_set[, ham_spam_SMS_freq_words]
ham_spam_SMS_test_set_freq10 <- ham_spam_SMS_test_set[, ham_spam_SMS_freq_words]
str(ham_spam_SMS_training_set_freq10)
str(ham_spam_SMS_test_set_freq10)

convert_counts <- function(x) {x <- ifelse(x > 0, "YES", "NO")}

ham_spam_SMS_train <- apply(ham_spam_SMS_training_set_freq10, MARGIN = 2, convert_counts)
ham_spam_SMS_test <- apply(ham_spam_SMS_test_set_freq10, MARGIN = 2, convert_counts)
str(ham_spam_SMS_train)
str(ham_spam_SMS_test)

#Naive Bayes Classifier
install.packages("e1071")
library(e1071)
ham_spam_NB_ModelClassifier <- naiveBayes(ham_spam_SMS_train, ham_spam_SMS_training_set_Labels)

#Prediction
ham_spam_NB_Predict <- predict(ham_spam_NB_ModelClassifier, ham_spam_SMS_test)


install.packages("gmodels")
library(gmodels)
CrossTable(ham_spam_NB_Predict, ham_spam_SMS_test_set_Labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('NB Prediction', 'Actual'))


ham_spam_NB_ModelClassifier_L1 <- naiveBayes(ham_spam_SMS_train, ham_spam_SMS_training_set_Labels, laplace = 1)

#Prediction
ham_spam_NB_Predict_L1 <- predict(ham_spam_NB_ModelClassifier_L1, ham_spam_SMS_test)
CrossTable(ham_spam_NB_Predict_L1, ham_spam_SMS_test_set_Labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('NB Prediction', 'Actual'))


ham_spam_NB_ModelClassifier_L2 <- naiveBayes(ham_spam_SMS_train, ham_spam_SMS_training_set_Labels, laplace = 2)
ham_spam_NB_Predict_L2 <- predict(ham_spam_NB_ModelClassifier_L1, ham_spam_SMS_test)
CrossTable(ham_spam_NB_Predict_L2, ham_spam_SMS_test_set_Labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('NB Prediction', 'Actual'))
