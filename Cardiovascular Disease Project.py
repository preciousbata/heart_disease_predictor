#!/usr/bin/env python
# coding: utf-8

# # Creating a Cardiovascular Disease Classification App and deployment in Heroku
# 
# _Machine Learning Program | Deployment_

# ## Step 1: Setting up the notebook
# We will be importing necessary libraries
# In[1]:
import numpy as np
import pandas as pd
import pickle
import logging

# ## Step 2: Preparing Logging
logging.basicConfig(filename='training_log_file.log',level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s  %(message)s')
logger=logging.getLogger()

# In[2]:
# Read in dataset
df=pd.read_csv('cardio_train.csv',sep=';',index_col='id')
logger.info('read in file')


# In[3]:
# Data Pre-Processing
X =df.drop('cardio',axis=1)
y=df['cardio']


# convert age from days to years
age_year=X['age'].apply(lambda x:x/365)
X['age']=np.ceil(age_year)
print(X)


# Now that we've transformed the data, we will select the top 5 features for our model.

from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)
Feature_Importance = pd.Series(model.feature_importances_,index=X.columns)
best_feat=Feature_Importance.nlargest(5)
feat_index=best_feat.index
X=X.filter(items=list(feat_index),axis=1)
X=X.astype('int')
print(X.shape)

# Model Fitting
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
logging.info('Model fitting')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=20)
clf=RandomForestClassifier(n_estimators=34)
# In[12]:
clf.fit(X_train,y_train)
#
# accuracy score
train_score=clf.score(X_train,y_train)
logging.info('the accuracy score for the the training is {}'.format(train_score))
# from sklearn.metrics import classification_report,confusion_matrix
#
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# # In[13]:
# pickle.dump(clf,open('clf_model.pkl','wb'))

