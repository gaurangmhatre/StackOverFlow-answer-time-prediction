
# coding: utf-8

# In[1]:

#from _future_ import division
import pandas as pd
import numpy as np
import scipy as sp
import math 
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.linear_model import LogisticRegression,Ridge,HuberRegressor,BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import time
import datetime


# In[2]:

inputFile = pd.read_csv('answers.csv')


# In[3]:

len(inputFile)


# In[4]:

#df = inputFile[:-100]
alldf= inputFile[:-1]
alldf=alldf.sort_values('at', ascending=True)
traindf = inputFile[26355:263540]
#train2=inputFile[263540:-1]
#traindf=traindf.append(train2)
traindf=traindf.sort_values('at', ascending=True)
print len(traindf)

testdf = inputFile[0:26354]
testdf=testdf.sort_values('at', ascending=True)
#263541
print len(testdf)


# In[5]:


def getData(data):
	tags=[]
	times=[]
	qids=[]
	data=data.drop_duplicates(subset='qid',keep='first')
	for index ,row in data.iterrows():
		qids.append(row['qid'])
		tags.append(getTags(row['tags']))
		times.append(int(row['at'])-int(row['qt']))
	return tags,times,qids


# In[6]:


def getTags(tags):
	tagList=tags.split(',')
	result=""
	for x in tagList:
		result+=str(x)+' '
	return result


# In[7]:

def writePredictions(predictions,actualTimes):
	correctCount=0
	length=len(predictions)
	file = open('Results/SVM/iteration1.txt', 'a')
	for x in range(len(predictions)):
		file.write(str(predictions[x])+" : "+str(actualTimes[x]))
		file.write("\n")
		diff=math.fabs(float(predictions[x]-actualTimes[x]))
		print str((diff/actualTimes[x]))
		if((diff/actualTimes[x])<0.20):
			correctCount=correctCount+1
	print correctCount
	print length		
	
	print str((correctCount/length)*100)


# In[8]:

testTags,testTimes,testQids=getData(testdf)
featureTags,answerTimes,qids=getData(traindf)


# In[9]:

alltags=set()
for index,row in alldf.iterrows():
	for feat in  row['tags'].split(','):
		alltags.add(feat)

#print alltags


# In[10]:

def getLinearPredictions(trainX,trainY,testX):
	regr = linear_model.LinearRegression()
	regr.fit(trainX,trainY)
	predictions=regr.predict(testX)
	return predictions


# In[11]:

def getLogisticPredictions(trainX,trainY,testX):
	regr = linear_model.LogisticRegression()
	regr.fit(trainX,trainY)
	predictions=regr.predict(testX)
	return predictions


# In[12]:

def getSVR(trainX,trainY,testX,choice):
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	if(choice==1):
		return svr_rbf.fit(trainX, trainY).predict(testX)
	if(choice==2):	
		return svr_lin.fit(trainX, trainY).predict(testX)
	if(choice==3):	
		return svr_poly.fit(trainX, trainY).predict(testX)


# In[13]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(norm='l2', vocabulary=list(alltags))
trainingMatrix=tf.fit_transform(featureTags)
testMatrix=tf.fit_transform(testTags)
print testMatrix[:-1]


# In[14]:

predictions=getSVR(trainingMatrix,answerTimes,testMatrix,1)




# In[15]:

#writeSVRPred(predictions,testTimes)
writePredictions(predictions,testTimes)





