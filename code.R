setwd("d:/Dropbox/SYNCHRO_POLSL/pml_proj")
set.seed(01021990)
library(caret)

data<-read.csv('pml-training.csv', na.strings=c("NA",""), strip.white=T)
summary(data)
# quite a few observations, first idea would be to remove rows with NAs, but then we are left with only few obserwation from class A
#so remove columns with Na's
isNA <- apply(data, 2, function(x) { sum(is.na(x)) })
data <- subset(data[, which(isNA == 0)])
# other useless variables: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window
data<-data[,-c(1,2,3,4,5,6,7)]
#data<-data[,-c(X, user_name, new_window, num_window, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)]

# check whether features are any good
nearZeroVar(data,saveMetrics=TRUE) #just plot those
#nsv <- nearZeroVar(data,saveMetrics=FALSE)
# thankfully no nsv to get rid of

#create data partitions:
inTrain <- createDataPartition(data$classe, p=0.7, list=F)
training <- data[inTrain,]
testing <- data[-inTrain,]

# first try: decisions trees
mdl<-train(classe~., data=training, method="rpart")
res1<-predict(mdl, training) #error on training set

t<-table(training$classe,res1)
sum(diag(t))/sum(t) #49.6%

res2<-predict(mdl, newdata=testing) #error on testing set
t<-table(testing$classe,res2)
sum(diag(t))/sum(t) #49.4%
# ok...so here the accuracy is just awful...

# use random forests and test feature importance
ctrl <- trainControl(allowParallel=T, method="cv", number=4) #constrain time-consuming bootstrapping
mdl1<-train(classe~., data=training, method="rf", trControl=ctrl)

#training set:
res4<-predict(mdl1, newdata=training) #error on training set set
confusionMatrix(training$classe, res4)$table
confusionMatrix(training$classe, res4)$overall[1] #crazy 100% accuracy :)

# testing set
res3<-predict(mdl1, newdata=testing) #error on testing set set
t<-table(testing$classe,res3)
sum(diag(t))/sum(t) #99.45% accuracy, so OUT OF SAMPLE ERROR is quite small here

# you know what else would be cool? look for best variables
varImp(mdl1)
# but I'm not going to use that today

# now...just do the honors :)
evaluation<-read.csv('pml-testing.csv')
evaluation <- subset(evaluation[, which(isNA == 0)])
evaluation<-evaluation[,-c(1,2,3,4,5,6,7)]
res_final<-predict(mdl1, newdata=evaluation) # performed well on those...got 20/20
res_final

# write files
n = length(res_final)
for(i in 1:n){
  filename = paste0("problem_id_",i,".txt")
  write.table(res_final[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
