"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    # List columns that should be dropped
    to_drop = ['Order No',
               'User Id',
               'Rider Id',
               'Vehicle Type',
               'Confirmation - Day of Month',
               'Confirmation - Weekday (Mo = 1)',
               'Confirmation - Time',
               'Arrival at Pickup - Day of Month',
               'Arrival at Pickup - Weekday (Mo = 1)',
               'Arrival at Pickup - Time',
               'Pickup - Day of Month',
               'Pickup - Weekday (Mo = 1)'
               'Platform Type']

    # drop columns
    feature_vector_df.drop(to_drop, axis = 1, inplace = True)

    # add features
    # add waiting time
    feature_vector_df['Pickup - Time'] = pd.to_datetime(feature_vector_df['Pickup - Time'])
    feature_vector_df['Placement - Time'] = pd.to_datetime(feature_vector_df['Placement - Time'])
    feature_vector_df['Waiting time'] = feature_vector_df['Pickup - Time'] - feature_vector_df['Placement - Time']
    feature_vector_df['Waiting time'] = feature_vector_df['Waiting time'].astype('timedelta64[s]')

    # convert day of month to cyclic feature
    feature_vector_df['month_sin'] = np.sin((feature_vector_df['Placement - Day of Month']-1)*(2.*np.pi/30.5))
    feature_vector_df['month_cos'] = np.cos((feature_vector_df['Placement - Day of Month']-1)*(2.*np.pi/30.5))

    # Converting the weekday to a cyclic feature
    feature_vector_df['day_sin'] = np.sin((feature_vector_df['Placement - Weekday (Mo = 1)']-1)*(2.*np.pi/7))
    feature_vector_df['day_cos'] = np.cos((feature_vector_df['Placement - Weekday (Mo = 1)']-1)*(2.*np.pi/7))

    # Converting the pickup time to date_time format and extracting the hour
    feature_vector_df['Pickup - Time'] = pd.to_datetime(feature_vector_df['Pickup - Time']).dt.hour
    # convert pickup time to cyclic feature
    feature_vector_df['hr_sin'] = np.sin(feature_vector_df['Pickup - Time']*(2.*np.pi/24))
    feature_vector_df['hr_cos'] = np.cos(feature_vector_df['Pickup - Time']*(2.*np.pi/24))

    # drop unecessary columns
    drop = ['Placement - Time', 'Pickup - Time','Placement - Weekday (Mo = 1)', 'Placement - Day of Month']
    feature_vector_df.drop(drop, axis = 1, inplace = True)

    # Impute missing values
    feature_vector_df['Precipitation in millimeters'].fillna(0,inplace=True)
    feature_vector_df['Temperature'].fillna(23.25,inplace=True)

    # Encode Personal or Business column
    feature_vector_df.loc[feature_vector_df['Personal or Business'] == 'Personal', 'Personal or Business'] = 1
    feature_vector_df.loc[feature_vector_df['Personal or Business'] == 'Business', 'Personal or Business'] = 0

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
