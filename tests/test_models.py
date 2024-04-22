import numpy as np 
import pandas as pd
import xgboost as xgb
import os
import pickle
import pytest
from unittest.mock import MagicMock,patch, mock_open
from src.models import XGBoostClassifier  # Import your XGBoostClassifier class
from src.models import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
import unittest
import builtins

def test_train_final_model_xgb():
    """
    Test training the final XGBoost model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X_train = pd.DataFrame(X_data)
    y_train= pd.DataFrame(np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]))


    model_trainer = XGBoostClassifier() # Initialize the class with the necessary attributes

    # Set mock values for best_params_, num_class, and random_state
    model_trainer.best_params_ = {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1}
    model_trainer.num_class = 6  
    model_trainer.random_state = 42  


    with unittest.mock.patch('builtins.open', return_value=MagicMock()) as mock_open: # Mock the open function to return a MagicMock object

        with unittest.mock.patch('pickle.dump') as mock_pickle_dump:  # Mock the pickle.dump function

            model_trainer.train_final_model(X_train, y_train) # Train the final model

            assert isinstance(model_trainer.final_model, xgb.XGBClassifier) # Check if the final model is an instance of XGBClassifier

            assert hasattr(model_trainer.final_model, 'classes_') # Check if the final model is trained

            expected_path = './src/models/xgboost_model.pkl' # Check if the model file is saved
            assert os.path.exists(expected_path)


def test_evaluate_model_xgb():
    """
    Test evaluating the XGBoost model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X_test = pd.DataFrame(X_data)
    y_test= pd.DataFrame(np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]))

    model = XGBoostClassifier() # Initialize the class with the necessary attributes

    with patch.object(model, 'save_metrics') as mock_save_metrics:
        file_path = './src/models/xgboost_model.pkl'

        model.evaluate_model(X_test, y_test,file_path) # Call the evaluate_model method
        
        assert isinstance(model.final_model, xgb.XGBClassifier)  # Assert that the final_model attribute is set correctly



def test_predict_xgb():
    """
    Test predicting with the XGBoost model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X = pd.DataFrame(X_data)

    model = RandomForestClassifier() # Initialize the class with the necessary attributes

    with patch.object(model, 'save_metrics') as mock_save_metrics:
        file_path = './src/models/xgboost_model.pkl'
        
        model.predict(X, file_path) # Call the predict method
        
        assert isinstance(model.final_model, xgb.XGBClassifier) # Assert that the final_model attribute is set correctly


def test_train_final_model_random_forest():
    """
    Test training the final Random Forest model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X_train = pd.DataFrame(X_data)
    y_train= pd.DataFrame(np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]))

    model_trainer = RandomForestClassifier() # Initialize the class with the necessary attributes

    # Set mock values for best_params_, num_class, and random_state
    model_trainer.best_params_ = {'n_estimators': 100, 'max_depth': 10}
    model_trainer.random_state = 42 

    with patch('builtins.open', return_value=MagicMock()) as mock_open: # Mock the open function to return a MagicMock object

        with patch('pickle.dump') as mock_pickle_dump: # Mock the pickle.dump function
        
            model_trainer.train_final_model(X_train, y_train) # Train the final model

            assert isinstance(model_trainer.final_model, SklearnRandomForestClassifier) # Check if the final model is an instance of RandomForestClassifier

            assert hasattr(model_trainer.final_model, 'classes_') # Check if the final model is trained

            expected_path = './src/models/randomforest_model.pkl' # Check if the model file is saved
            assert os.path.exists(expected_path)


def test_evaluate_model_random_forest():
    """
    Test evaluating the Random Forest model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X_test = pd.DataFrame(X_data)
    y_test= pd.DataFrame(np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]))

    model = RandomForestClassifier() # Initialize the class with the necessary attributes

    with patch.object(model, 'save_metrics') as mock_save_metrics:
        file_path = './src/models/randomforest_model.pkl'

        model.evaluate_model(X_test, y_test, file_path) # Call the evaluate_model method
        
        assert isinstance(model.final_model, SklearnRandomForestClassifier) # Assert that the final_model attribute is set correctly

def test_predict_random_forest():
    """
    Test predicting with the Random Forest model.
    """
    # Create mock data (not used in this test)
    X_data = np.random.rand(10, 83)
    X_data[:, 0] = np.random.randint(0, 5, 10)
    X = pd.DataFrame(X_data)

    model = RandomForestClassifier()  # Initialize the class with the necessary attributes

    with patch.object(model, 'save_metrics') as mock_save_metrics:
        file_path = './src/models/randomforest_model.pkl'
    
        model.predict(X, file_path) # Call the evaluate_model method
        
        assert isinstance(model.final_model, SklearnRandomForestClassifier) # Assert that the final_model attribute is set correctly