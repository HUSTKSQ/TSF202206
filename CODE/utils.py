'''
Descripttion: 
version: 
Author: Shenqiang Ke
Date: 2022-06-16 22:30:18
LastEditors: Please set LastEditors
LastEditTime: 2022-06-16 23:25:24
'''
from pyexpat import model
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from models import *

def get_label(df_needs, is_train = False):
    
    df_needs['date']  = df_needs['date'].map(lambda x: x[:-3]).values
    if is_train:
        agg_dict = {'label':'sum',
               'is_sale_day':{'sum','count'},
              }
    
    else:
        agg_dict = {'is_sale_day':{'sum','count'}}
    
    df_label = df_needs.groupby(['product_id','date']).agg(agg_dict)
    df_label.columns = ['_'.join(x)  for x in df_label.columns]
    df_label = df_label.reset_index()
    return df_label


def get_features(df_orders):
    df_orders['date'] = df_orders['year'].astype(str) + '-' + df_orders['month'].map(lambda x:'0'+str(x) if x<=9 else str(x))
    
    # 该类型在该月的销量总量
    df_orders['type_order_sum'] = df_orders.groupby(['type','date'])['order'].transform('sum').values
    # 销量在该类型中的比例
    df_orders['order_ratio'] = df_orders['order'].values/ df_orders['type_order_sum'].values
    
    df_orders['stock_diff'] = df_orders['end_stock'].values - df_orders['start_stock'].values
    df_orders['type_stock_diff_sum'] = df_orders.groupby(['type','date'])['stock_diff'].transform('sum').values
    # stock_diff在该类型中的比例
    df_orders['stock_diff_ratio'] = df_orders['order'].values/ df_orders['type_order_sum'].values
    
    # 删除year
    del df_orders['year']
    
    # lag特征
    for i in range(3):
        df_orders[f'stock_diff_ratio_{i}'] = df_orders.groupby('product_id')['stock_diff_ratio'].shift(i).values
        df_orders[f'stock_diff_{i}']       = df_orders.groupby('product_id')['stock_diff'].shift(i).values
        df_orders[f'order_ratio_{i}']      = df_orders.groupby('product_id')['order_ratio'].shift(i).values
        df_orders[f'order_{i}']            = df_orders.groupby('product_id')['order'].shift(i).values
    
    return df_orders

def get_dataset(root):
    df_orders_train = pd.read_csv(os.path.join('/tmp', root, '商品月订单训练集.csv'))
    df_orders_test = pd.read_csv(os.path.join('/tmp', root, '商品月订单测试集.csv'))

    df_needs_train = pd.read_csv(os.path.join('/tmp', root, '商品需求训练集.csv'))
    df_needs_test = pd.read_csv(os.path.join('/tmp', root, '商品需求测试集.csv'))

    df_train_label = get_label(df_needs_train,is_train = True)
    df_test_label  = get_label(df_needs_test,is_train = False)

    df_orders_train_fea = get_features(df_orders_train)
    df_orders_test_fea  = get_features(df_orders_test)

    df_train = df_train_label.merge(df_orders_train_fea, on = ['product_id','date'], how = 'left')
    df_test  = df_test_label.merge(df_orders_test_fea, on = ['product_id','date'], how = 'left')

    type_dict = {"A1":1,"A2":2,"A3":3}
    df_train['type'] = df_train['type'].map(type_dict).values
    df_test['type']  = df_test['type'].map(type_dict).values

    features = [c for c in df_test.columns if c not in ['date','label_sum']]
    # label = 'label_sum'

    train = df_train.copy()

    # test = df_test[features]

    return train, df_test, features

def get_dataloader(train, features,  val=True, batch_size=16):

    # 把所有含nan的行全部删除
    data_train = train.dropna(axis=0, how='any')

    train_data = data_train[features]
    train_label = data_train['label_sum']

    train_data = np.array(train_data)
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.array(train_label).reshape(-1)

    train_dataset = TensorDataset(torch.tensor(train_data.astype(float),dtype=float), torch.tensor(train_label.astype(float)))

    if val:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        data_train, data_val = random_split(train_dataset, [train_size, val_size])
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    if val:
        dataloaders['val'] = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
        
    return dataloaders

def get_model_ml(model_name):
    model_name = model_name.lower()
    if model_name == 'lgb':
        return lgb_reg()
        