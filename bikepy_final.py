###########Library###########

import os

import pandas as pd

import matplotlib.pyplot as pt

import numpy as np

os.chdir("C:/Users/welcome/Desktop/Project-2")

os.getcwd()

data=pd.read_csv("day.csv")

data.dtypes

data.isnull().sum()

########Dropping the unimportant variable#####

data=data.drop("casual",axis=1)

data=data.drop("registered",axis=1)

data=data.drop("instant",axis=1)

data=data.drop("dteday",axis=1)

data.shape

###########Checking for missing values############

data.isnull().sum()

#############Changing into proper datatype#######

num=['season','yr','mnth','holiday','weekday','workingday','weathersit']

for i in num:
    data.loc[:,i]=data.loc[:,i].astype("object")
    
    data.dtypes
    
data['yr']=np.where(data['yr']==0,'2011','2012')

data['season']=np.where(data['season']==1,'spring',
    np.where(data['season']==2,'summer',np.where(data['season']==3,'fall',
             'winter')))
data['mnth']=np.where(data['mnth']==1,'Jan',
    np.where(data['mnth']==2,'feb',
             np.where(data['mnth']==3,'mar',
                      np.where(data['mnth']==4,'apr',
                               np.where(data['mnth']==5,'may',
                                        np.where(data['mnth']==6,'Jun',
                                                 np.where(data['mnth']==7,'Jul',
                                                          np.where(data['mnth']==8,'aug',
                                                                   np.where(data['mnth']==9,'sep',
                                                                            np.where(data['mnth']==10,'oct',
                                                                                     np.where(data['mnth']==11,'Nov','Dec')))))))))))
                                                                                              

    
###############visualisation############
 
##########Histogram for numeric attribute###########
    
pt.hist(data.temp,bins=20,color='red')

pt.xlabel('Temperature')

pt.ylabel('Frequency')

pt.title('Histogram of Temperature')
###########
pt.hist(data.atemp,bins=20,color='blue')

pt.xlabel('Actual Temperature')

pt.ylabel('Frequency')

pt.title('Histogram of Actual Temperature')
########
pt.hist(data.windspeed,bins=20,color='grey')

pt.xlabel('Windspeed')

pt.ylabel('Frequency')

pt.title('Histogram of Windspeed')
###########
pt.hist(data.hum,bins=20,color='yellow')

pt.xlabel('Humidity')

pt.ylabel('Frequency')

pt.title('Histogram of Humidity')

############Scatterplot for numeric attibute#########

pt.scatter(x=data.cnt,y=data.temp,color='red')

pt.ylabel('Temperature')

pt.xlabel('Bike Count')

pt.title('Bike Count vs Temperature')
#########
pt.scatter(x=data.cnt,y=data.atemp,color='green')

pt.ylabel('Actual Temperature')

pt.xlabel('Bike Count')

pt.title('Bike Count vs  Actual Temperature')
########
pt.scatter(x=data.cnt,y=data.windspeed,color='blue')

pt.ylabel('Windspeed')

pt.xlabel('Bike Count')

pt.title('Bike Count vs Windspeed')
#########
pt.scatter(x=data.cnt,y=data.hum,color='Yellow')

pt.ylabel('Humidity')

pt.xlabel('Bike Count')

pt.title('Bike Count vs Humidity')

##############Barplot for Factor Variable########

num=['season','yr','mnth','holiday','weekday','workingday','weathersit']

import seaborn as sn

sn.barplot(x="season",y="cnt",data=data)

sn.barplot(x="yr",y="cnt",data=data)

sn.barplot(x="mnth",y="cnt",data=data)

sn.barplot(x="holiday",y="cnt",data=data)

sn.barplot(x="weekday",y="cnt",data=data)

sn.barplot(x="workingday",y="cnt",data=data)

sn.barplot(x="weathersit",y="cnt",data=data)

##########Boxplot for numeric variable######

sn.boxplot(y=data.temp)

sn.boxplot(y=data.atemp)

sn.boxplot(y=data.hum)

sn.boxplot(y=data.windspeed)

sn.boxplot(y=data.cnt)


data=data.drop(data[(data.cnt <100)].index,axis=0)

###########Remove outliers######
data.dtypes
no1=['temp','atemp','windspeed','hum','cnt']
for i in no1:
    q75,q25=np.percentile(data.loc[:,i],[75,25])
    iqr=q75-q25
    mi=q25-(1.5*iqr)
    ma=q75+(1.5*iqr)
    data.loc[data.loc[:,i]<mi,:i]=np.nan
    data.loc[data.loc[:,i]>ma,:i]=np.nan

data.isnull().sum()

data=data.dropna()

###############Correlation#############

cor=data.loc[:,no1]

co_mat=cor.corr().round(2)

data=data.drop('atemp',axis=1)

#########Dummies creation##########

num=['season','yr','mnth','holiday','weekday','workingday','weathersit']

for i in num:
    tem=pd.get_dummies(data[i],prefix=i)
    data=data.join(tem)
    
data.dtypes

data=data.drop('season',axis=1)
   
data=data.drop('yr',axis=1)
    
data=data.drop('mnth',axis=1)
   
data=data.drop('holiday',axis=1)

data=data.drop('weekday',axis=1)

data=data.drop('workingday',axis=1)

data=data.drop('weathersit',axis=1)

#############Sampling############

x=data.drop('cnt',axis=1)

y=data.iloc[:,3].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

x_tr=x_train

x_te=x_test

###################Linear model###########

import statsmodels.api as sm

mod_1=sm.OLS(y_train,x_train).fit()

mod_1.summary()

max(mod_1.pvalues)

x_train=x_train.drop('mnth_Jul',axis=1)

mod_2=sm.OLS(y_train,x_train).fit()

mod_2.summary()

max(mod_2.pvalues)

x_train=x_train.drop('mnth_feb',axis=1)

mod_3=sm.OLS(y_train,x_train).fit()

mod_3.summary()

max(mod_3.pvalues)

x_train=x_train.drop('mnth_Jan',axis=1)

mod_4=sm.OLS(y_train,x_train).fit()

mod_4.summary()

max(mod_4.pvalues)

x_train=x_train.drop('weekday_1.0',axis=1)

mod_5=sm.OLS(y_train,x_train).fit()

mod_5.summary()

max(mod_5.pvalues)

x_train=x_train.drop('weekday_2.0',axis=1)

mod_6=sm.OLS(y_train,x_train).fit()

mod_6.summary()

max(mod_6.pvalues)

x_train=x_train.drop('weekday_4.0',axis=1)

mod_7=sm.OLS(y_train,x_train).fit()

mod_7.summary()

max(mod_7.pvalues)

x_train=x_train.drop('weekday_3.0',axis=1)

mod_8=sm.OLS(y_train,x_train).fit()

mod_8.summary()

max(mod_8.pvalues)

x_train=x_train.drop('mnth_Dec',axis=1)

mod_9=sm.OLS(y_train,x_train).fit()

mod_9.summary()

max(mod_9.pvalues)

x_train=x_train.drop('mnth_Nov',axis=1)

mod_10=sm.OLS(y_train,x_train).fit()

mod_10.summary()

max(mod_10.pvalues)

x_train=x_train.drop('weekday_5.0',axis=1)

mod_11=sm.OLS(y_train,x_train).fit()

mod_11.summary()

max(mod_11.pvalues)

x_train.columns


x_test=x_test.loc[:,['temp', 'hum', 'windspeed', 'season_fall', 'season_spring',
       'season_summer', 'season_winter', 'yr_2011', 'yr_2012', 'mnth_Jun',
       'mnth_apr', 'mnth_aug', 'mnth_mar', 'mnth_may', 'mnth_oct', 'mnth_sep',
       'holiday_0.0', 'holiday_1.0', 'weekday_0.0', 'weekday_6.0',
       'workingday_0.0', 'workingday_1.0', 'weathersit_1.0', 'weathersit_2.0',
       'weathersit_3.0']]

pr1=mod_11.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score

rmse=np.sqrt(mean_squared_error(y_test,pr1))

r2=(r2_score(y_test,pr1))

def mape_error(acu_val,pred_val):
    mape=np.mean(np.abs((acu_val-pred_val)/acu_val))*100
    return mape

mape_error(y_test,pr1)

######Rmse=738
######MAPE=16.10%
######Accuracy=83.90%
####r2=86.66%

################Decision tree###########

from sklearn import tree 

tree_mod=tree.DecisionTreeRegressor(random_state=0).fit(x_tr,y_train)

pr3=tree_mod.predict(x_te)

rmse1=np.sqrt(mean_squared_error(y_test,pr3))

mape_error(y_test,pr3)

r2_1=(r2_score(y_test,pr3))

######Rmse=941
######MAPE=19.78%
######Accuracy=80.22%
####r2=78.37%

################Random Forest###########

from sklearn.ensemble import RandomForestRegressor

for_mod=RandomForestRegressor().fit(x_tr,y_train)

pr2=for_mod.predict(x_te)

rmse2=np.sqrt(mean_squared_error(y_test,pr2))

mape_error(y_test,pr2)

r2_2=(r2_score(y_test,pr2))


######Rmse=680
######MAPE=14.83%
######Accuracy=85.17%
##########r2=88.70%


##############XGboost########

import xgboost

xg=xgboost.XGBRegressor(n_estimators=100,learning_rate=0.05,gamma=0,subsample=0.50,
                        colsample_bytree=1,max_depth=4).fit(x_tr,y_train)

pr3=xg.predict(x_te)

rmse3=np.sqrt(mean_squared_error(y_test,pr3))

mape_error(y_test,pr3)

r2_3=(r2_score(y_test,pr3))

######Rmse=617.93
######MAPE=13.52%
######Accuracy=86.48%
##########r2=90.68%

x_te['cnt']=pr3

x_te.to_csv("output for sample data in python .csv")















































































































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    