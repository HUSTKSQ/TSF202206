'''
Descripttion: 
version: 
Author: Shenqiang Ke
Date: 2022-06-16 22:35:15
LastEditors: Please set LastEditors
LastEditTime: 2022-06-16 23:30:11
'''

import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--is_machinelearning', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='lgb')
parser.add_argument('--is_HPO', type=bool, default=True)
parser.add_argument('--submit_title', type=str, default='0616Test')
args = parser.parse_args()

# Data Read
root = '/home/keshenqiang/KDXF/DATA'

train, test, features = get_dataset(root)

# Machines Learning Methods
if args.is_machinelearning:
    X_train = train[features]
    y_train= train['label_sum']

    model = get_model_ml(args.model_name)
    model.fit(X_train, y_train)
    if args.is_HPO:
        model = model.best_estimator_
    
    pred = model.predict(test[features])

else:
    dataloaders = get_dataloader(train, features)



# Result Submitting
# Prerequisition: the type of pred is nmupyarray and the shape is (627, )
test['label'] = pred
df_submit = test[['date','product_id','label']].copy()
df_submit.rename(columns = {'date':'month'},inplace = True)
df_submit['label'] = df_submit['label'].map(lambda x: 0 if x <0 else x)
df_submit[['month','product_id','label']].to_csv(f'{args.submit_title}.csv',index = None)