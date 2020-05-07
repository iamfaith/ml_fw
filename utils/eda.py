##    https://tianchi.aliyun.com/notebook-ai/detail?postId=95276

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


train_df = pd.read_csv('datalab/231784/used_car_train_20200313.csv', sep=' ')
print(train_df.shape)
train_df.describe()


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure()
sns.distplot(train_df['price'])
plt.figure()
train_df['price'].plot.box()
plt.show()


import gc


test_df = pd.read_csv('datalab/231784/used_car_testA_20200313.csv', sep=' ')
print(test_df.shape)
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
del train_df, test_df
gc.collect()
df.head()



#-----------
date_cols = ['regDate', 'creatDate']
cate_cols = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
num_cols = ['power', 'kilometer'] + ['v_{}'.format(i) for i in range(15)]
cols = date_cols + cate_cols + num_cols

tmp = pd.DataFrame()
tmp['count'] = df[cols].count().values
tmp['missing_rate'] = (df.shape[0] - tmp['count']) / df.shape[0]
tmp['nunique'] = df[cols].nunique().values
tmp['max_value_counts'] = [df[f].value_counts().values[0] for f in cols]
tmp['max_value_counts_prop'] = tmp['max_value_counts'] / df.shape[0]
tmp['max_value_counts_value'] = [df[f].value_counts().index[0] for f in cols]
tmp.index = cols
tmp

#--
from tqdm import tqdm


def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


for f in tqdm(date_cols):
    df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
    df[f + '_year'] = df[f].dt.year
    df[f + '_month'] = df[f].dt.month
    df[f + '_day'] = df[f].dt.day
    df[f + '_dayofweek'] = df[f].dt.dayofweek
    
    
    
#----
plt.figure()
plt.figure(figsize=(16, 6))
i = 1
for f in date_cols:
    for col in ['year', 'month', 'day', 'dayofweek']:
        plt.subplot(2, 4, i)
        i += 1
        v = df[f + '_' + col].value_counts()
        fig = sns.barplot(x=v.index, y=v.values)
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(f + '_' + col)
plt.tight_layout()
plt.show()


corr1 = abs(df[~df['price'].isnull()][['price'] + date_cols + num_cols].corr())
plt.figure(figsize=(10, 10))
sns.heatmap(corr1, linewidths=0.1, cmap=sns.cm.rocket_r)



#------
from scipy.stats import entropy


feat_cols = []

### count编码
for f in tqdm([
    'regDate', 'creatDate', 'regDate_year',
    'model', 'brand', 'regionCode'
]):
    df[f + '_count'] = df[f].map(df[f].value_counts())
    feat_cols.append(f + '_count')

### 用数值特征对类别特征做统计刻画，随便挑了几个跟price相关性最高的匿名特征
for f1 in tqdm(['model', 'brand', 'regionCode']):
    g = df.groupby(f1, as_index=False)
    for f2 in tqdm(['v_0', 'v_3', 'v_8', 'v_12']):
        feat = g[f2].agg({
            '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
            '{}_{}_median'.format(f1, f2): 'median', '{}_{}_mean'.format(f1, f2): 'mean',
            '{}_{}_std'.format(f1, f2): 'std', '{}_{}_mad'.format(f1, f2): 'mad'
        })
        df = df.merge(feat, on=f1, how='left')
        feat_list = list(feat)
        feat_list.remove(f1)
        feat_cols.extend(feat_list)

### 类别特征的二阶交叉
for f_pair in tqdm([
    ['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']
]):
    ### 共现次数
    df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['SaleID'].transform('count')
    ### n unique、熵
    df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
        '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
        '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
    }), on=f_pair[0], how='left')
    df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
        '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
        '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
    }), on=f_pair[1], how='left')
    ### 比例偏好
    df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
    df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
    
    feat_cols.extend([
        '_'.join(f_pair) + '_count',
        '{}_{}_nunique'.format(f_pair[0], f_pair[1]), '{}_{}_ent'.format(f_pair[0], f_pair[1]),
        '{}_{}_nunique'.format(f_pair[1], f_pair[0]), '{}_{}_ent'.format(f_pair[1], f_pair[0]),
        '{}_in_{}_prop'.format(f_pair[0], f_pair[1]), '{}_in_{}_prop'.format(f_pair[1], f_pair[0])
    ])