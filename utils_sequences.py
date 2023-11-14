import os
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler


def get_data(data_dir, files, lookback = 90, test_prop = 0.1):
    # The scaler objects will be stored in this dictionary so that our output test data from the model can be re-scaled during evaluation
    label_scalers = {}

    train_x = []
    test_x = {}
    test_y = {}

    for file in tqdm_notebook(files): 
        # Store csv file in a Pandas DataFrame
        df = pd.read_csv(data_dir + file, parse_dates=[0])
        # Processing the time data into suitable input formats
        df['hour'] = df.apply(lambda x: x['Datetime'].hour,axis=1)
        df['dayofweek'] = df.apply(lambda x: x['Datetime'].dayofweek,axis=1)
        df['month'] = df.apply(lambda x: x['Datetime'].month,axis=1)
        df['dayofyear'] = df.apply(lambda x: x['Datetime'].dayofyear,axis=1)
        df = df.sort_values("Datetime").drop("Datetime",axis=1)
        
        # Scaling the input data
        sc = MinMaxScaler()
        label_sc = MinMaxScaler()
        data = sc.fit_transform(df.values)
        # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
        label_sc.fit(df.iloc[:,0].values.reshape(-1,1))
        label_scalers[file] = label_sc
        
        # Define lookback period and split inputs/labels
        #lookback = 90
        inputs = np.zeros((len(data)-lookback,lookback,df.shape[1]))
        labels = np.zeros(len(data)-lookback)
        
        for i in range(lookback, len(data)):
            inputs[i-lookback] = data[i-lookback:i]
            labels[i-lookback] = data[i,0]
        inputs = inputs.reshape(-1,lookback,df.shape[1])
        labels = labels.reshape(-1,1)
        
        # Split data into train/test portions and combining all data from different files into a single array
        test_portion = int(test_prop*len(inputs))
        if len(train_x) == 0:
            train_x = inputs[:-test_portion]
            train_y = labels[:-test_portion]
        else:
            train_x = np.concatenate((train_x,inputs[:-test_portion]))
            train_y = np.concatenate((train_y,labels[:-test_portion]))
        test_x[file] = (inputs[-test_portion:])
        test_y[file] = (labels[-test_portion:])

    return train_x, test_x, train_y, test_y, label_scalers