import pandas as pd

mc_df = pd.read_json(r'C:\Users\86189\PycharmProjects\MLproject\Machine-Learning\train_data_all.json')
#print(mc_df.head())
print(mc_df.columns)
print(mc_df.info())
missing_data = pd.DataFrame({'total_missing': mc_df.isnull().sum(), 'perc_missing': (mc_df.isnull().sum()/87766)*100})
#print(missing_data)