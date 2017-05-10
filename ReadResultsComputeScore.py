
# coding: utf-8

# In[1]:

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
import math 


# In[2]:




# In[341]:

#Clustering iteration1
#with open("Results/Clustering/iteration10.txt", "r") as fh:
#    lines = fh.readlines() 

#Clustering iteration2
#with open("Results/SVM/iteration10.txt", "r") as fh:
with open("Results/LinearRegression/iteration10.txt", "r") as fh:
#with open("Results/LogisticRegression/iteration10.txt", "r") as fh:
    lines = fh.readlines() 


# In[342]:

print len(lines)


# In[343]:

def getListOfPredictionsAndActualResults(lines):
    actualResults = []
    predictions = []
    
    for line in lines:
        #print line
        
        result = line.strip('\n').split(" : ")
        predictions.append(result[0])
        actualResults.append(result[1])
        
        
        #print str(actualResults) + "   ----  "+ str(predictions)
        
    return actualResults,predictions 


# In[344]:

actualResults, predictions =   getListOfPredictionsAndActualResults(lines)


# In[345]:

print str(len(actualResults))
print str(len(predictions))


# In[346]:

def getScores(actualResults,predictions):
    print "F1 Score"
    #print f1_score(actualResults, predictions, average='macro')  
    #print f1_score(actualResults, predictions, average='micro')  
    print f1_score(actualResults, predictions, average='weighted')  
    
    print "Recall"
    #print recall_score(actualResults, predictions, average='macro')
    #print recall_score(actualResults, predictions, average='micro')
    print recall_score(actualResults, predictions, average='weighted')
    
    print "Precision"
    #print precision_score(actualResults, predictions, average='macro')  
    print precision_score(actualResults, predictions, average='micro')  
    #print precision_score(actualResults, predictions, average='weighted')  
    
    print "mean_squared_error"
    print mean_squared_error(actualResults, predictions)
    


# In[347]:

def getTestData(actualResults,predictions):
    actualTestList = []
    predictionTestList = []
    
    for x in range(len(actualResults)):
        diff=math.fabs(float(predictions[x])-float(actualResults[x]))
        
        #print diff
        #print float(actualResults[x])
        #print diff/float(actualResults[x])
        
        
        if float(actualResults[x])!=0 and (diff/float(actualResults[x]))<0.30:
            predictionTestList.append(1)
        else:
            predictionTestList.append(0)
        actualTestList.append(1)
        #break
    return actualTestList, predictionTestList
    


# In[348]:

actualtestList, predictionTestList = getTestData(actualResults,predictions)
getScores(actualtestList,predictionTestList)





