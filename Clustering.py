
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy as sp

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import time
import datetime
import math


# In[2]:

inputFile = pd.read_csv('answers.csv')
df = inputFile[:26354]


# In[3]:

#Iteration1
#First 263540 rows
#testdf = inputFile[0:26354]

#Iteration2
#Second 263540 rows
#testdf = inputFile[26355:52708]


#Iteration3
#Third 263540 rows
#testdf = inputFile[52709:79062]

#Iteration4
#Forth 263540 rows
#testdf = inputFile[79063:105416]

#Iteration5
#Fifth 263540 rows
#testdf = inputFile[105417:131770]

#Iteration6
#Sixth 263540 rows
#testdf = inputFile[131771:158124]

#Iteration7
#Seventh 263540 rows
#testdf = inputFile[158125:184478]

#Iteration8
#Eighth 263540 rows
#testdf = inputFile[184479:210832]

#Iteration9
#Ninth 263540 rows
#testdf = inputFile[210833:237186]

#Iteration10
#Tenth 263540 rows
testdf = inputFile[237187:263540]


# In[4]:

print len(testdf)
print(len(df))


# In[5]:

# Look at the first 3 rows
df[:3]


# Columns values,
# qid: Unique question id
# 
# i: User id of questioner
# 
# qs: Score of the question
# 
# qt: Time of the question (in epoch time)
# 
# tags: a comma-separated list of the tags associated with the question. Examples of tags are ``html'', ``R'', ``mysql'', ``python'', and so on; often between two and six tags are used on each question.
# 
# qvc: Number of views of this question (at the time of the datadump)
# 
# qac: Number of answers for this question (at the time of the datadump)
# 
# aid: Unique answer id
# 
# j: User id of answerer
# 
# as: Score of the answer
# 
# at: Time of the answer

# In[6]:

df['tags'][:10]


# In[7]:

features =  set()

def feature_selection(data):
    for rows in data:
        for feat in  rows.split(','):
            features.add(feat)
            
            
                        


# In[295]:

feature_selection(df['tags']) # Get list of all the unique tags in features
feature_selection(testdf['tags'])
print(len(features))


# In[296]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(norm='l2', vocabulary=list(features))


# In[297]:

training_matrix =  tf.fit_transform(df['tags'])
training_feature_names = tf.get_feature_names() 


# In[298]:

test_matrix =  tf.fit_transform(testdf['tags'])


# In[299]:

#print(training_feature_names)


# In[300]:

training_matrix[:2]


# In[301]:

X=training_matrix
y = test_matrix


# In[302]:

kmeans = KMeans(n_clusters=500, random_state=0).fit(X)


# In[303]:

predication = []

predication = kmeans.predict(y)

len(predication)
#print(predication)


# In[304]:

print(kmeans.cluster_centers_[1])


# In[305]:

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)


# In[306]:

##Closest node from the centroids
print len(closest)  


# In[307]:

predications= []
predications= predication

print(predications)


# In[308]:

PredicationTime = []
ActualTime  = []
for predication in predications:
    closestNode = (closest[predication])
    answerTime = (df.at[closestNode, 'at']) - (df.at[closestNode, 'qt'])  
    actualTimeForTest =  (df.at[predication,'at']) - (df.at[predication,'qt'])
    
    PredicationTime.append(answerTime) 
    ActualTime.append(actualTimeForTest)


# In[309]:

print len(ActualTime)
print len(PredicationTime)


# In[ ]:




# In[311]:

def writePredictions(predictions,actualTimes):
    correctCount=0
    length=len(predictions)
    file = open('Results/Clustering/iteration10.txt', 'a')
    #file = open('Results/result.dat', 'a')
    for x in range(len(predictions)):
        file.write(str(predictions[x])+" : "+str(actualTimes[x]))
        file.write("\n")
        diff=math.fabs(float(predictions[x]-actualTimes[x]))
        #print str((diff/actualTimes[x]))
        if((diff/actualTimes[x])<0.20):
            correctCount=correctCount+1
    print correctCount
    print length

    print str((correctCount/length)*100)


# In[312]:

writePredictions(PredicationTime,ActualTime)


# In[ ]:




# In[ ]:



