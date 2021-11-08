# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

X = pd.read_csv('mobile_price_X.csv')
y = pd.read_csv('mobile_price_y.csv')


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from xgboost import XGBClassifier
xg = XGBClassifier(booster='gbtree', gamma=0, learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1,
              n_estimators=100, n_jobs=-1, predictor='auto',
              random_state=10, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=None,subsample=1, tree_method='exact',
              use_label_encoder=True, verbosity=None)

#Fitting model with trainig data
xg.fit(X.values, y.values)

# Saving model to disk
pickle.dump(xg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))