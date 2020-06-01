"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


# Fetch training data and preprocess for modeling
train_raw = pd.read_csv('data/Train.csv') 
riders_raw = pd.read_csv('data/Riders.csv')
train = train_raw.merge(riders_raw, how="left", on="Rider Id")

ys_train = train[['Time from Pickup to Arrival']]
Xs_train = train[['Pickup Lat','Pickup Long',
                 'Destination Lat','Destination Long']]

# Fit model
model = RandomForestRegressor(n_estimators=500, random_state=101, max_depth=10, max_leaf_nodes=200)
print ("Training Model...")
model.fit(Xs_train.reshape(-1,1), ys_train) #added .reshape on this line


# Pickle model for use within our API
save_path = '../trained-models/rfmodel.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model, open(save_path,'wb'))