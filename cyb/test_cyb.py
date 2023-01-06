import pandas as pd
from collections import Counter
import math
import numpy as np
#统计数据库数据分布情况
# mc_df = pd.read_json(r'C:\Users\86189\PycharmProjects\MLproject\Machine-Learning\train_data_all.json')
# #print(mc_df.head())
# print(mc_df.columns)
# print(mc_df.info())
# missing_data = pd.DataFrame({'total_missing': mc_df.isnull().sum(), 'perc_missing': (mc_df.isnull().sum()/87766)*100})
#print(missing_data)

############################################
   #贝叶斯回归

def delete_len1(list):
   for i in range(len(list) - 1, -1, -1):
      if len(list[i]) == 1 and not list[i].isdigit():
         del list[i]
def split2word(words):
   word = []
   for i in range(len(words)):
      word += words[i].split()
   delete_len1(word)
   return word

# for i in range(len(set)-1,-1,-1):#倒叙删除
#    if set[i].startswith('"'):
#       set[i] += set[i+1]
#       del(set[i+1])
def set_divide(set,data):
   #分出训练集和测试集,一个元素是一个代表一个sample的字符串
   # test_set = set[:3000]
   # train_set = set[3000:]
   #得到测试集和训练集
   testnum1 = 0
   testnum2 = 0
   testnum3 = 0
   test_keyset = []
   temp_keyset = []
   temp1_keyset = []
   train_keyset = []
   for i in range(len(set)):
      #words = train_set[i].split(',')#按逗号拆每行
      if data.iat[i,2] == "True to Size":
         if testnum2 < 300:
            test_keyset.append(i)
         else:
            temp_keyset.append(i)
         testnum2 += 1
      if data.iat[i,2] == "Small":
         if testnum1 < 300:
            test_keyset.append(i)
         else:
            temp1_keyset.append(i)
         testnum1 += 1
      if data.iat[i,2] == "Large":
         if testnum3 < 300:
            test_keyset.append(i)
         else:
            temp1_keyset.append(i)
         testnum3 += 1
   temp_keyset = np.array(temp_keyset)
   train_keyset = np.array_split(temp_keyset,5)
   a = np.array(temp1_keyset)
   for i in range(5):
      train_keyset[i] = np.append(train_keyset[i],a)
   return test_keyset,train_keyset
   # else:
   #    train_keyset.append(i)
   # if line.__contains__("True to Size") and testnum2 < 1000:
   #    test_set.append(line)
   #    testnum2 += 1
   # if line.__contains__("Small") and testnum1 < 1000:
   #    test_set.append(line)
   #    testnum1 += 1
   # if line.__contains__("Large") and testnum3 < 1000:
   #    test_set.append(line)
   #    testnum3 += 1
   # else:
   #    train_set.append(line)
# print(set[1])
# print(test_set[1])
# print("len of testset is {}".format(len(test_keyset)))
# print("shape of trainset is {}".format(len(train_keyset)))
def train(set,train_keyset):
   #将训练集划分到三个分类中
   T2S_set = []#True to Size
   S_set = []#Small
   L_set = []#Large
   T2S_num = 0
   S_num = 0
   L_num = 0
   #print(train_keyset)
   for i in train_keyset:
      words = set[i].split(',')#按逗号拆每行
      if data.iat[i,2] == "True to Size":
         word = split2word(words)
         T2S_set += word
         T2S_num += 1
      if data.iat[i,2] == "Small":
         word = split2word(words)
         S_set += word
         S_num += 1
      if data.iat[i,2] == "Large":
         word = split2word(words)
         L_set += word
         L_num += 1
   #print(T2S_num,S_num,L_num)
   #训练过程
   V_set = list(dict.fromkeys(T2S_set + S_set + L_set))
   V = len(V_set)
   D = len(train_keyset)
   #true to size
   P_t2s = T2S_num/D
   n_t2s = len(T2S_set)
   n_t2s_dict = Counter(T2S_set)
   P_k_t2s = {}
   for word , value in n_t2s_dict.items():
      value = (value + 1)/(n_t2s + V)
      P_k_t2s[word] = value
   #small
   P_s = S_num/D
   n_s = len(S_set)
   n_s_dict = Counter(S_set)
   P_k_s = {}
   for word , value in n_s_dict.items():
      value = (value + 1)/(n_s + V)
      P_k_s[word] = value
   #large
   P_l = L_num/D
   n_l = len(L_set)
   n_l_dict = Counter(L_set)
   P_k_l = {}
   for word , value in n_l_dict.items():
      value = (value + 1)/(n_l + V)
      P_k_l[word] = value
   P_list = [P_s,P_t2s,P_l]
   P_k_list = []
   P_k_list.append(P_k_s)
   P_k_list.append(P_k_t2s)
   P_k_list.append(P_k_l)
   return P_list, P_k_list

def Predict(word_list, P_list, P_k_list):#之后在每个if分支中插入Predict判断结果
   predict_s = 0
   predict_t2s = 0
   predict_l = 0
   for j in range(5):
      y_t2s = math.log10(P_list[j][1])
      y_s = math.log10(P_list[j][0])
      y_l = math.log10(P_list[j][2])
      #添加直接比较五个预测器数量
      for i in range(len(word_list)):
         # if spam_train.__contains__(word_list[i]):
         temp1 = P_k_list[j][0].get(word_list[i], 1)
         temp2 = P_k_list[j][1].get(word_list[i], 1)
         temp3 = P_k_list[j][2].get(word_list[i], 1)
         if temp1 == 1 or temp2 == 1 or temp3 ==1: continue
         y_s += math.log10(temp1)
         y_t2s += math.log10(temp2)
         y_l += math.log10(temp3)
         if y_s >= y_t2s:
            predict_s += 1
         elif y_t2s >= y_l:
            predict_t2s += 1
         else:
            predict_l += 1
   predict = 0
   if predict_s >= predict_t2s:
      predict = 1
   elif predict_t2s >= predict_l:
      predict = 2
   else:
      predict = 3
   return predict-1

def test(set, test_keyset):
   #test
   confusion_matrix = [[0,0,0], [0,0,0], [0,0,0]]#预测值，真实值，第二元素是预测为2，真实为1
   for i in test_keyset:
      #words = train_set[i].split(',')#按逗号拆每行
      line= set[i]
      both = 0
      if data.iat[i,2] == "True to Size":
         line = line.split(',')
         line.remove("True to Size")
         word_list = split2word(line)
         confusion_matrix[2-1][Predict(word_list, P_list, P_k_list)] += 1
         both += 1
      if data.iat[i,2] == "Small":
         line = line.split(',')
         line.remove("Small")
         word_list = split2word(line)
         confusion_matrix[1-1][Predict(word_list, P_list, P_k_list)] += 1
         both += 1
      if data.iat[i,2] == "Large":
         line = line.split(',')
         line.remove("Large")
         word_list = split2word(line)
         confusion_matrix[3-1][Predict(word_list, P_list, P_k_list)] += 1
         both += 1
      # if data.iat[i, 5] == "Large":
      #    line = line.replace('Large', '')
      #    word_list = split2word(line.split(','))
      #    confusion_matrix[3-1][Predict(word_list, 3, y_t2s, y_s, y_l)] += 1
      #    both += 1
      # if both > 1:
      #    print(set[i])
   return confusion_matrix

set = []
f = open(r"C:/Users/86189/PycharmProjects/MLproject/Machine-Learning/cyb/data_proc.txt",'r',encoding="utf-8")
set = f.readlines()
del(set [0])
f.close()
data = pd.read_csv('data_proc.txt')
test_keyset, train_keyset = set_divide(set, data)
P_list = []
P_k_list = []
for i in range(5):
   x, y = train(set,train_keyset[i])
   P_list.append(x)
   P_k_list.append(y)
#print(P_list)
confusion_matrix = test(set,test_keyset)
accuracy_s = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0])
recall_s = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2])
accuracy_t2s = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])
recall_t2s = confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2])
accuracy_l = confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])
recall_l = confusion_matrix[2][2]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2])
print("predict_s is {}".format(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0]))
print("predict_t2s is {}".format((confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])))
print("predict_l is {}".format((confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])))
print("accuracy_s is {}".format(accuracy_s))
print("recall_s is {}".format(recall_s))
print("accuracy_t2s is {}".format(accuracy_t2s))
print("recall_t2s is {}".format(recall_t2s))
print("accuracy_l is {}".format(accuracy_l))
print("recall_l is {}".format(recall_l))

#result
# accuracy_s is 0.3125
# recall_s is 0.014204545454545454
# accuracy_t2s is 0.7329302987197724
# recall_t2s is 0.9951714147754708
# accuracy_l is 0.2
# recall_l is 0.0024390243902439024

#predict_s is 31
# predict_t2s is 2946
# predict_l is 23
# accuracy_s is 0.6129032258064516
# recall_s is 0.019
# accuracy_t2s is 0.3363883231500339
# recall_t2s is 0.991
# accuracy_l is 0.8260869565217391
# recall_l is 0.019

#'review_summary','review','rating','fit'
# predict_s is 225
# predict_t2s is 2751
# predict_l is 24
# accuracy_s is 0.6622222222222223
# recall_s is 0.149
# accuracy_t2s is 0.35986913849509267
# recall_t2s is 0.99
# accuracy_l is 0.75
# recall_l is 0.018

#5 predictor contain all attributes
# predict_s is 497
# predict_t2s is 292
# predict_l is 111
# accuracy_s is 0.5070422535211268
# recall_s is 0.84
# accuracy_t2s is 0.6883561643835616
# recall_t2s is 0.67
# accuracy_l is 0.4954954954954955
# recall_l is 0.18333333333333332

# 5predictors test attributes
# predict_s is 708
# predict_t2s is 115
# predict_l is 77
# accuracy_s is 0.3432203389830508
# recall_s is 0.81
# accuracy_t2s is 0.3826086956521739
# recall_t2s is 0.14666666666666667
# accuracy_l is 0.35064935064935066
# recall_l is 0.09
