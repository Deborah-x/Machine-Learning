'''
`author` xxx PB20061256
`date` 2022.12.5
'''

import os, sys
import re
import json
import pandas as pd
import numpy as np  
import collections
import math
import random


vocab=set()
total_dataset=[]
train_dataset={1:[],2:[],3:[]}
train_post={1:{},2:{},3:{}}

test_dataset_labeled={1:[],2:[],3:[]}
test_dataset_predict={1:[],2:[],3:[]}

sample_num={1:0,2:0,3:0}
prior={1:0.0,2:0.0,3:0.0}

label_types=int(3)
train_ratio=float(0.9)
word_length_threhold=4

#delete all non-english word tokens
def cleanText(string:str)->str: 
	string=string.replace("Subject","",1) #delete the header of mail	
	r="[,./;:\[\]\'\"-=_+~!@#$%^&*()`\\|?><]"
	string = re.sub(r, ' ', string)
	string = re.sub(r"\s{2,}", " ", string)
	r = "[^A-Za-z ]"
	string = re.sub(r, '', string)

	return string.strip().lower()

#split the str into word list
def getWordList(text:str)->list:
	word_list=text.split(' ')
	idx=0
	while idx < len(word_list):
		word=word_list[idx]
		if len(word)<=2:
			word_list.remove(word)
		else:
			idx+=1
	return word_list

def getLabel(fit_tag:str)->int:
	if fit_tag== "Small":
		return 1
	elif fit_tag== "True to Size":
		return 2
	elif fit_tag== "Large":
		return 3
	else:
		return 4  # for missing target tag:"fit"

def preprocessData(data:dict):
  label=getLabel(data['fit'])
  review_corpus=cleanText(data['review']+data['review_summary'])
  review_word_list=getWordList(review_corpus)
  if (label<=label_types and len(review_word_list)>word_length_threhold): # delete these too-short reviews 
    return label,review_word_list
  else:
    return None,None

def preprocess(file_name:str="./train_data_all.json"):
  raw_dataset = json.load(open(file_name, encoding="utf-8"))  # class list
  for data in raw_dataset:
    newLabel,newWordList=preprocessData(data)
    if (newLabel!=None):
      total_dataset.append({'label':newLabel,'wordList':newWordList})

  total_data_count=len(total_dataset)
  #randomize the train set and dataset every time
  random.shuffle(total_dataset)

  #seperate the dataset into train set and test set by ratio of train_ratio
  for idx in range(0,math.floor(total_data_count*train_ratio)):
    new_label=total_dataset[idx]['label']
    new_wordList=total_dataset[idx]['wordList']
    train_dataset[new_label]+=new_wordList
    sample_num[new_label]+=1
	
  for idx in range(math.floor(total_data_count*train_ratio)+1,total_data_count):
    new_label=total_dataset[idx]['label']
    test_dataset_labeled[new_label].append(idx)

def trainNB():
  global vocab
  vocabNum={1:None,2:None,3:None}
  word_num={1:0,2:0,3:0}
  total_sample_num=0

  for idx in range(1,label_types+1):
    vocabNum[idx]=(collections.Counter(train_dataset[idx]))
    word_num[idx]=len(train_dataset[idx])
    total_sample_num+=sample_num[idx]
  vocab=set(train_dataset[1]+train_dataset[2]+train_dataset[3])
  vocab_num=len(vocab)

  for idx in range(1,label_types+1):
    prior[idx]=float(sample_num[idx])/total_sample_num
    for word in vocab:
      train_post[idx][word]=(vocabNum[idx][word]+1)/(vocab_num+word_num[idx])
  

def testNB():
  global vocab
  log_prob={1:0,2:0,3:0}
  new_log_prob={1:0,2:0,3:0}
  for idx in range(1,label_types+1):
    log_prob[idx]=math.log(prior[idx])

  total_data_count=len(total_dataset)
  for data_idx in range(math.floor(total_data_count*train_ratio)+1,total_data_count):
    new_data=total_dataset[data_idx]['wordList']
    words=[]
    for word in new_data:
      if word in vocab:
        words.append(word)
    for idx in range(1,label_types+1):
      new_log_prob[idx]=log_prob[idx]
    for word in words:
      for idx in range(1,label_types+1):
        new_log_prob[idx]+=math.log(train_post[idx][word])
    pred_label=max(new_log_prob,key=new_log_prob.get)
    test_dataset_predict[pred_label].append(data_idx)

def evaluateNB():
  correctMat=np.zeros((3,3))
  for row in range(0,label_types):
    for col in range(0,label_types):
      correctMat[row][col]=len(set(test_dataset_labeled[row+1])&set(test_dataset_predict[col+1]))/len(test_dataset_labeled[row+1])
  print("total dataset size:",len(total_dataset))
  print("test dataset size:",math.floor(len(total_dataset)*(1-train_ratio)))
  print("The correct ratio:\n",correctMat)

def main():
	preprocess()
	trainNB()
	testNB()
	evaluateNB()
	exit(0)

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.getcwd()))
	print(f"Current working directory: {os.getcwd()}")
	main()
