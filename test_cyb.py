import pandas as pd
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

#分出训练集和测试集
test_set = set[:5000]
train_set = set[5000:]

def delete_not_alpha(list):
   for i in range(len(list) - 1, -1, -1):
      if not list[i].isalpha() or len(list[i]) == 1:
         del list[i]

#将训练集划分到三个分类中
T2S_set = []#True to Size
S_set = []#Small
L_set = []#Large
for i in range(len(train_set)):
   words = train_set[i].split()#需要自己写一个函数来执行拆词的功能，split不行
   if words.__contains__("True to Size"):
      T2S_set += words
   if words.__contains__("Small"):
      S_set += words
   if words.__contains__("Large"):
      L_set += words
#需要自己写一个函数来执行拆词的功能
# print(len(T2S_set))
# print(len(S_set))
# print(len(L_set))
print(train_set[0])
print(train_set[0].split())