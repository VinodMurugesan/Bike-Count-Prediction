#######Bike count prediction##########
rm(list=ls())
getwd()
data=read.csv("day.csv",header=T)
dim(data)
str(data)
names(data)
sum(is.na(data))
sum(duplicated(data$instant))
str(data)
############Changing variable to factor#######
names(data)
num=c('season','yr','mnth','holiday','weekday','workingday','weathersit')
for(i in num){
  data[,i]=as.factor(data[,i])
}
str(data)
################Visualisation#############
library(ggplot2)
##############factor data########
fac=c('season','yr','mnth','holiday','weekday','workingday','weathersit')
for(i in (1:length(fac))){
  assign(paste0('f',i),ggplot(data,aes_string(x=fac[i],y='cnt'))+geom_bar(stat='identity',fill='darkslateblue')+xlab(fac[i])+ylab("Frequency")+
    ggtitle("Bar plot of",fac[i])+theme_bw()+theme(text=element_text(size = 15)))
}
gridExtra::grid.arrange(f1,f2,ncol=2)
gridExtra::grid.arrange(f3,f4,ncol=2)
gridExtra::grid.arrange(f5,f6,f7,ncol=3)

#########Scatter plot########
no=c('temp','atemp','hum','windspeed')
for(i in (1:length(no))){
  assign(paste0('n',i), ggplot(data,aes_string(x=no[i],y='cnt'))+geom_point(color='green')+xlab(no[i])+ylab("Bike count")+
    ggtitle("scatterplot of",no[i])+theme_bw()+theme(text=element_text(size = 15)))
}
gridExtra::grid.arrange(n1,n2,ncol=2)
gridExtra::grid.arrange(n3,n4,ncol=2)


###########Histogram###########
no=c('temp','atemp','hum','windspeed')
for(i in (1:length(no))){
  assign(paste0('h',i), ggplot(data,aes_string(x=no[i]))+geom_histogram(color='Brown',bins=20)+xlab(no[i])+ylab("Frequency")+
           ggtitle("Histogram of",no[i])+theme_bw()+theme(text=element_text(size = 15))+geom_density())
}
gridExtra::grid.arrange(h1,h2,ncol=2)
gridExtra::grid.arrange(h3,h4,ncol=2)

###################Boxplot#############
no=c('temp','atemp','hum','windspeed')
for(i in (1:length(no))){
  assign(paste0('b',i), ggplot(data,aes_string(y=no[i]))+geom_boxplot(color='red',fill='orange')+xlab(no[i])+ylab("Frequency")+
           ggtitle("boxplot of",no[i])+theme_bw()+theme(text=element_text(size = 15)))
}
gridExtra::grid.arrange(b1,b2,ncol=2)
gridExtra::grid.arrange(b3,b4,ncol=2)

######Outlier Detection and Removal############
no=c('temp','atemp','hum','windspeed')
for (i in no)
{
  box=data[,i][data[,i]%in% boxplot.stats(data[,i])$out] 
  data[,i][data[,i]%in% box]=NA
}
data$cnt=ifelse(data$cnt>100,data$cnt,NA)
sum(is.na(data))
data=na.omit(data)
#######correlation#############
library(corrgram)
cor(data[,no])
corrgram(data[,no],order=F,upper.panel = panel.pie,text.panel = panel.text,main="correlation plot")
names(data)
data=data[,-11]
##############Creating Dummies###########
str(data)
d1=data.frame(model.matrix(~season,data))
d1=d1[,-1]
d2=data.frame(model.matrix(~mnth,data))
d2=d2[,-1]
d3=data.frame(model.matrix(~weekday,data))
d3=d3[,-1]
d4=data.frame(model.matrix(~weathersit,data))
d4=d4[,-1]
dim(data)
str(data)
data=data[,-c(1:2)]
str(data)
names(data)
data=data[,-c(11,12)]
dim(data)
data=cbind(data,d1,d2,d3,d4)
str(data)
data=data[,-c(1,3,5,7)]
##############Sample #############
library(caret)
sam=createDataPartition(data$cnt,p=0.80,list=F)
train=data[sam,]
test=data[-sam,]
names(test)
################Linear Regression#############
lm1=lm(cnt~.,data)
summary(lm1)
library(car)
vif(lm1)
library(MASS)
step=stepAIC(lm1)
lm2=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 + mnth3 + mnth4 + mnth5 + mnth6 + mnth8 + mnth9 + 
         mnth10 + weekday1 + weekday6 + weathersit2 + weathersit3,data)
summary(lm2)
vif(lm2)
##################Month4#################
lm3=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
     season4 + mnth3 + mnth5 + mnth6 + mnth8 + mnth9 + 
     mnth10 + weekday1 + weekday6 + weathersit2 + weathersit3,data)
summary(lm3)
vif(lm3)
##################Month6#################
lm4=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 + mnth3 + mnth5 + mnth8 + mnth9 + 
         mnth10 + weekday1 + weekday6 + weathersit2 + weathersit3,data)
summary(lm4)
vif(lm4)
##################Month5#################
lm5=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 + mnth3 + mnth8 + mnth9 + 
         mnth10 + weekday1 + weekday6 + weathersit2 + weathersit3,data)
summary(lm5)
##################Weekday1#################
lm6=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 + mnth3 + mnth8 + mnth9 + 
         mnth10 +  weekday6 + weathersit2 + weathersit3,data)
##################Month8#################
lm7=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 + mnth3 + mnth9 + 
         mnth10 +  weekday6 + weathersit2 + weathersit3,data)
summary(lm7)
##################Month3#################
lm8=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + season3 + 
         season4 +  mnth9 + 
         mnth10 +  weekday6 + weathersit2 + weathersit3,data)
summary(lm8)
##################Season3#################
lm9=lm(cnt ~ yr + workingday + temp + hum + windspeed + season2 + 
         season4 + mnth9 + 
         mnth10 +  weekday6 + weathersit2 + weathersit3,data)
summary(lm9)
vif(lm9)
names(test)
pr=predict(lm9,test[,-7])
library(DMwR)
regr.eval(test$cnt,pr,stats = c('rmse','mape'))
####Accuracy=83.35%
####RMSE=735.02
####MAPE=16.65%
################Decision Tree#########
library(rpart)
tree_mod=rpart(cnt~.,train,method="anova")
summary(tree_mod)
pre1=predict(tree_mod,test[,-7])
regr.eval(test$cnt,pre1,stat=c("rmse","mape"))
####Accuracy=78.8%
####RMSE=851.36
####MAPE=21.20%
###############Random Forest############
library(randomForest)
forest_mod=randomForest(cnt~.,train,importance=T,ntree=100)
summary(forest_mod)
pre2=predict(forest_mod,test[,-7])
regr.eval(test$cnt,pre2,stat=c("rmse","mape"))
####Accuracy=83.76%
####RMSE=633.14
####MAPE=16.24%
####################SVR#################
library(e1071)
svr_mod=svm(cnt~.,data,type='eps-regression')
summary(svr_mod)
pre3=predict(svr_mod,test[,-7])
out=cbind(test[,-7],pre3)
write.csv(out,"output for sample data.csv")
regr.eval(test$cnt,pre3,stat=c("rmse","mape"))
####Accuracy=88.89%
####RMSE=457.67
####MAPE=11.11%
#################XGboost###########
library(caret)
control=trainControl(method='cv',number=5,savePredictions = T,classProbs = T)
paragrid=expand.grid(eta=0.1,gamma=1,max_depth=3,nrounds=100,colsample_bytree=0.7,
                     min_child_weight=2,subsample=0.5)
model=train(cnt~.,data=train,method='xgbTree',Control=control,tuneGrid=paragrid)
pre5=predict(model,test[,-7])
regr.eval(test$cnt,pre5,stat=c("rmse","mape"))
####Accuracy=87.0%
####RMSE=594.50
####MAPE=13.02%




