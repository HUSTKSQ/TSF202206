'''
Descripttion: 
version: 
Author: Shenqiang Ke
Date: 2022-06-14 10:17:41
LastEditors: Please set LastEditors
LastEditTime: 2022-06-14 19:09:49
'''
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")

from PredictionModel import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='arima', help='Optional: arima, prophet, holt')
args = parser.parse_args()


# Parameter setting
path = '/home/keshenqiang/KDXF/DATA/商品需求训练集.csv'

# model_name = 'prophet'

future = 3
insample = False

# Model setting
model_name = args.model_name.lower()
if model_name == "holt":
    train_params = {}
    pred_params = {}
elif model_name == "arima":
    train_params = {'m': 12}
    pred_params = {}
elif model_name == "prophet":
    train_params = {"growth": "linear", "seasonality_mode": "multiplicative", "cap": 10000}
    pred_params = {}

# Data Read
data = pd.read_csv(path, sep=',',header=0)

product_id = list(data['product_id'].unique())

month_list = ['2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',\
    '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
    '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12']

didx = pd.date_range("2018-02-01", periods = 35, freq ='M')

total_data = []
id_zero = {}
for id in product_id:
    # print(id)
    data_id = data[data['product_id']==id]
    data_id['date'] = pd.to_datetime(data_id['date'])
    data_id = data_id.set_index('date') 

    month_label = []
    for month in month_list:
        try:
            month_label.append(data_id[month]['label'].sum())
            if data_id[month]['label'].sum() == 0.:
                print(f'The sales volume of Product {id} in {month} was 0!')
                if id in id_zero.keys():
                    id_zero[id] += 1
                else:
                    id_zero[id] = 0
        except:
            month_label.append(0)
            print(f'The sales volume of Product {id} in {month} was unattainable!')
            if id not in id_zero.keys():
                id_zero[id] = 0
            else:
                id_zero[id] += 0
    total_data.append(month_label)

f = open(f'{model_name}.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(['month', 'product_id', 'label'])

#TODO Predict_result --> CSV File: The result writing order
result = []

for i in range(len(total_data)):
    id = product_id[i]
    print(id)
    print('*'*40)
    train_data = pd.DataFrame(total_data[i], index=didx)
    train_data.index.set_names = 'month'
    train_data = train_data[0]
    # print(train_data)
    
    # If there are only five or less five months's sales data of a product
    # Just simply consider the sales in the next three months as the sales in the last month given in the training data
    # if id in id_zero.keys() and id_zero[id] >= 30:    
    #     csv_writer.writerow(['2021-01', id, int(train_data[-1]) if int(train_data[-1])>0 else 0])
    #     csv_writer.writerow(['2021-02', id, int(train_data[-1]) if int(train_data[-1])>0 else 0])
    #     csv_writer.writerow(['2021-03', id, int(train_data[-1]) if int(train_data[-1])>0 else 0])
    # else:
    #     model = ModelDispatcher().dispatch(model_name=model_name, component='raw')
    #     model.set_parameters(train_data, train_params)

    #     predict_result = model.predict(future, pred_params)

    #     csv_writer.writerow(['2021-01', id, int(predict_result[0]) if int(predict_result[0])>0 else 0])
    #     csv_writer.writerow(['2021-02', id, int(predict_result[1]) if int(predict_result[1])>0 else 0])
    #     csv_writer.writerow(['2021-03', id, int(predict_result[2]) if int(predict_result[2])>0 else 0])
    
    model = ModelDispatcher().dispatch(model_name=model_name, component='raw')
    model.set_parameters(train_data, train_params)

    predict_result = model.predict(future, pred_params)

    predicts = []
    for j in predict_result:
        predicts.append(int(j) if j>0 else 0)

    result.append(predicts)

    # csv_writer.writerow(['2021-01', id, int(predict_result[0]) if int(predict_result[0])>0 else 0])
    # csv_writer.writerow(['2021-02', id, int(predict_result[1]) if int(predict_result[1])>0 else 0])
    # csv_writer.writerow(['2021-03', id, int(predict_result[2]) if int(predict_result[2])>0 else 0])

month_dir = ['2021-01', '2021-02', '2021-03']

for j in range(3):
    for i in range(len(product_id)):
        id = product_id[i]
        csv_writer.writerow([month_dir[j], id, result[i][j]])

f.close()