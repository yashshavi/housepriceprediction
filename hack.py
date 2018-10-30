import pandas as pd
import numpy as np
housing=pd.read_csv('hackathon_rentomter_nobroker.csv')
housing=housing.drop(['id'],axis=1)
amenities=pd.get_dummies(housing['amenities'],drop_first=True)
col = [cname for cname in housing.columns if  housing[cname].dtype == "object"]
data2=housing

for c in col:
    dummy=pd.get_dummies(data2[c],drop_first=True)
    data2=pd.concat([data2,dummy],axis=1)
for c in col:
    data2.drop([c],axis=1,inplace=True)
housing=data2
x=housing.drop(['rent'],axis=1)
y=housing['rent']
y= np.log1p(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet,Lasso
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#gbrt=GradientBoostingRegressor(random_state=0)
lm.fit(X_train,y_train)
fullprediction=lm.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,fullprediction)
print(score)