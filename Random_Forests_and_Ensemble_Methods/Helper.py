
# coding: utf-8

# In[2]:


from __future__ import print_function
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
import time
import random
from random import sample 
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree


# In[3]:


#scikit learn classifier with train values
def scikit_learn(train):
    y = train[train.columns[-1]].values.tolist()
    x = train[train.columns[:-1]].values.tolist()
    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(x,y)
    return clf
    


# In[4]:


#Prediction of scikit-learn on a given row. 
def scikit_predict(clf,test):
    x_test = test[test.columns[:-1]].values.tolist()
    y_test = test[test.columns[-1]].values.tolist()
    cnt = 0
    accurate = 0
    for index,row in enumerate(x_test):
        cnt+=1
        test = clf.predict([row])
        if(test[0]==y_test[index]): 
            accurate+=1
    return 1.0*accurate/cnt
        
    


# In[5]:


#Count of each row in the dataset. 
def class_counts(dataset):
    
    last_name = dataset.columns[-1]  
    last_col = dataset[last_name]
    counter = Counter(last_col) 
    return counter
    


# In[6]:



def most_frequent(counter): 
    maxi = 0
    ele = 0
    for i in counter.keys():
        if(counter[i]>maxi):
            ele = i
            maxi = counter[i]
    return ele


# In[7]:


def gini(rows):

    
    classes = class_counts(rows)
    
    total=len(rows)
    p=0
    for i in classes:
        p+= (classes[i]*1.0 / total)**2
        
        
        
    return 1-p


# In[8]:


def info_gain(upper, lower, current):
    upper_len = len(upper)
    lower_len = len(lower)
    p = lower_len*1.0 / (lower_len + upper_len)
    
    s = p * gini(lower) + (1 - p) * gini(upper)
    gain = current - s
    
    return gain


# In[9]:


class Output: 
    def __init__(self,rows):
        self.prediction = most_frequent(class_counts(rows))
    
        
   


# In[10]:


class Decision_Node:
    
    def __init__(self,cutoff,attr,true,false): 
        self.true = true
        self.false = false
        self.attr = attr
        self.cutoff = cutoff


# In[11]:


def best_split(dataset): 
    best_gain = -10
    best_attr = 0 
    best_cutoff = 0
    
    current = gini(dataset)
    
    features = dataset.columns[:-1]
    
    for attr in features:
        cutoff = set(dataset[attr])
        for cut in cutoff:
            upper,lower = part(dataset,cut,attr)
            if(len(upper)==0 or len(lower)==0): 
                continue
           
            gain = info_gain(upper,lower,current)
            if(gain>best_gain): 
                best_gain = gain
                best_attr = attr
                best_cutoff = cut
        
    return best_gain, best_attr, best_cutoff
            
            


# In[12]:


def part(dataset,cutoff,attr):
    
    upper = dataset[dataset[attr]>cutoff]
    lower = dataset[dataset[attr]<=cutoff]
    
    return upper,lower
    


# In[13]:


def build_tree(dataset,depth,greedy): 
    
    best_gain, best_attr,best_cutoff = best_split(dataset)
    if greedy:
        if(best_gain<=0):
            return Output(dataset)
    else: 
        if(depth==0 or len(dataset)<5): 
            return Output(dataset)
    
    upper,lower = part(dataset,best_cutoff,best_attr)
    
    true_branch = build_tree(upper,depth-1,greedy)
    false_branch = build_tree(lower,depth-1,greedy)
    
    return Decision_Node(best_cutoff,best_attr,true_branch,false_branch)
        


# In[14]:


#Check a given row for the condition.
def check(row,node):
    at = node.attr
    
    if(row[at]>node.cutoff):
        return True
    else:
        return False



def classify_iris(row,node):
    if isinstance(node,Output): 
        return node.prediction
    else: 
        if(check(row,node)):
            return classify_iris(row,node.true)
        else:
            return classify_iris(row,node.false)
    



def divide_data(dataset):
    attr = dataset.columns
    train = []
    test = []
    for index,row in dataset.iterrows():
        if(index%3==0):
            test.append(row)
        else: 
            train.append(row)
    return pd.DataFrame(train),pd.DataFrame(test)
    


def accuracy(test,tree):
    accurate = 0
    cnt = 0
    listed = test.values.tolist()
    for index,row in test.iterrows(): 
        cnt+=1
        if(row.tolist()[-1]==classify_iris(row,tree)):
            accurate+=1
    return 1.0*accurate/cnt
    


def random_features(dataset,m):
    features = dataset.columns[:-1].tolist()
    chosen_names = sample(features,m)
    chosen_names.append(dataset.columns[-1])

    return dataset[chosen_names]




def random_forest(dataset,m,n,depth): 
    tree_arr = []
    for i in range(n):
        tree_arr.append([])
        trimmed = random_features(dataset,m)
    
        tree_ = build_tree(trimmed,depth,False)
        tree_arr[i] = tree_
    return tree_arr




def test_random_forests(tree_arr,test):
    accuracy = 0
    cnt = 0
    for index,row in test.iterrows(): 
        output_array = []
        cnt+=1
        for node in tree_arr:
            decision = classify_iris(row,node)
            output_array.append(decision)
        prediction = most_common(output_array)
        if row.tolist()[-1] == prediction:
            accuracy+=1
    return 1.0*accuracy/cnt
        
    

def build_random_tree(dataset,depth,no_of_features):
    random_dataset = random_features(dataset,no_of_features)
    tree_random = build_tree(random_dataset,depth,False)
    return tree_random


def most_common(lst):
    return max(set(lst), key=lst.count)










    
    




