import pandas as pd
from collections import Counter
import math
#统计数据库数据分布情况
# mc_df = pd.read_json(r'C:\Users\86189\PycharmProjects\MLproject\Machine-Learning\train_data_all.json')
# #print(mc_df.head())
# print(mc_df.columns)
# print(mc_df.info())
# missing_data = pd.DataFrame({'total_missing': mc_df.isnull().sum(), 'perc_missing': (mc_df.isnull().sum()/87766)*100})
#print(missing_data)

############################################
   #贝叶斯回归


set = []
f = open(r"C:/Users/86189/PycharmProjects/MLproject/Machine-Learning/data_proc.txt",'r',encoding="utf-8")
set = f.readlines()
del(set [0])
f.close()

#品名被拆分成两行了，合并成一行
for i in range(len(set)-1,-1,-1):#倒叙删除
   if set[i].startswith('"'):
      set[i] += set[i+1]
      del(set[i+1])

#分出训练集和测试集,一个元素是一个代表一个sample的字符串
test_set = set[:3000]
train_set = set[3000:]

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

#将训练集划分到三个分类中
T2S_set = []#True to Size
S_set = []#Small
L_set = []#Large
T2S_num = 0
S_num = 0
L_num = 0
for i in range(len(train_set)):
   words = train_set[i].split(',')#按逗号拆每行
   if words.__contains__("True to Size"):
      word = split2word(words)
      T2S_set += word
      T2S_num += 1
   if words.__contains__("Small"):
      word = split2word(words)
      S_set += word
      S_num += 1
   if words.__contains__("Large"):
      word = split2word(words)
      L_set += word
      L_num += 1

#训练过程
V_set = list(dict.fromkeys(T2S_set + S_set + L_set))
V = len(V_set)
D = len(train_set)
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

#test
predict = 0 # 1 =small. 2 = true to size. 3 = large
confusion_matrix = [[0,0,0], [0,0,0], [0,0,0]]#预测值，真实值，第二元素是预测为2，真实为1
accuracy = 0
y_t2s = math.log10(P_t2s)
y_s = math.log10(P_s)
y_l = math.log10(P_l)
def Predict(word_list, label, y_t2s, y_s, y_l):#之后在每个if分支中插入Predict判断结果
   for i in range(len(word_list)):
      # if spam_train.__contains__(word_list[i]):
      temp1 = P_k_s.get(word_list[i], 1)
      temp2 = P_k_t2s.get(word_list[i], 1)
      temp3 = P_k_l.get(word_list[i], 1)
      if temp1 == 1 or temp2 == 1 or temp3 ==1: continue
      y_s += math.log10(temp1)
      y_t2s += math.log10(temp2)
      y_l += math.log10(temp3)
      if y_s >= y_t2s:
         predict = 1
      elif y_t2s >= y_l:
         predict = 2
      else:
         predict = 3
      return predict-1



for i in range(len(test_set)):
   #words = train_set[i].split(',')#按逗号拆每行
   line= test_set[i]
   if line.__contains__("True to Size"):
      line = line.replace('True to Size', '')
      word_list = split2word(line.split(','))
      confusion_matrix[2-1][Predict(word_list, 2, y_t2s, y_s, y_l)] += 1
   if line.__contains__("Small"):
      line = line.replace('Small', '')
      word_list = split2word(line.split(','))
      confusion_matrix[1-1][Predict(word_list, 1, y_t2s, y_s, y_l)] += 1
   if line.__contains__("Large"):
      line = line.replace('Large', '')
      word_list = split2word(line.split(','))
      confusion_matrix[3-1][Predict(word_list, 3, y_t2s, y_s, y_l)] += 1

accuracy_s = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0])
recall_s = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2])
accuracy_t2s = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])
recall_t2s = confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2])
accuracy_l = confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])
recall_l = confusion_matrix[2][2]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2])
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