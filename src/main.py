import argparse
import numpy as np
import pandas as pd
from data_load_and_preproc import DataProcessor
from data_covid_analyzer import DataAnalyzer
from models import XGBoostClassifier, RandomForestClassifier
from utils import train_test_split_df
from sklearn.preprocessing import LabelEncoder

def df_prep_for_model(df_tot):
    """
    Prepare DataFrame for model training and prediction.

    Parameters:
    - df_tot (pandas.DataFrame): DataFrame containing the data.

    Returns:
    - df (pandas.DataFrame): Processed DataFrame ready for model training.
    - df_tot (pandas.DataFrame): Original DataFrame with rows containing 'm' values dropped.
    """
    rows_dropped = len(df_tot[df_tot[83] == 'm']) # Count number of rows with 'm' values of column with labels
    df_tot.drop(df_tot[df_tot[83] == 'm'].index, inplace=True)  # Drop rows with 'm' values of column with labels

    df_tot2 = df_tot.copy()
    df_tot2[83] = df_tot2[83]-1 # Decrement the values in column 83 by 1
    df_tot2[83] = df_tot2[83].astype(int) # Convert column 83 to integer type
    label_encoder = LabelEncoder() # Initialize label encoder
    df_tot2[0] = label_encoder.fit_transform(df_tot2[0]) # Encode values in column 0

    df = df_tot2.iloc[:,1:]  # Extract features for model training

    return df,df_tot

def bool_argparse(arg):
    """
    Parse boolean argument from string.

    Parameters:
    - arg (str): String representation of boolean value.

    Returns:
    - bool: Parsed boolean value.
    """
    if arg == "True": # Check if argument is "True"
        return True
    elif arg == "False": # Check if argument is ""
        return False

def main(args):
    """
    Main function for model training and prediction.

    Parameters:
    - args (argparse.Namespace): Parsed command line arguments.
    """
    data_processor = DataProcessor(folder=args.dir) # Initialize DataProcessor object
    df_ls = data_processor.create_df()  # Create DataFrames from ARFF files
    for df in df_ls:
        df.iloc[:, 2:] = df.iloc[:, 2:].applymap(data_processor.convert_to_float)  # Convert values to float
    data_processor.check_unique_strings(df_ls) # Check for unique strings in DataFrames
    data_processor.check_specific_value(df_ls) # Check for specific value in DataFrames
    df_ls = data_processor.replace_m(df_ls) # Replace 'm' values in DataFrames


    data_analyzer = DataAnalyzer(df_ls) # Initialize DataAnalyzer object
    variations_all = data_analyzer.analyze_data() # Analyze data and get variations pre/post Covid

    df_tot = pd.concat(df_ls)
    df,df_tot = df_prep_for_model(df_tot) # Prepare DataFrame for model training

    if args.model == "xgb":
        model = XGBoostClassifier() # Initialize XGBoost model

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [200,300,400,500],
            'max_depth': [10,15,20,50],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
        }

    elif args.model == "rf":
        model = RandomForestClassifier()  # Initialize Random Forest model

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [10, 15, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

    else:
        raise ValueError("Invalid model option. Choose 'xgb' for XGBoost or 'rf' for Random Forest.")

    bool_training = bool_argparse(args.training) # Parse boolean training argument
    bool_test = bool_argparse(args.test) # Parse boolean test argument
    bool_predict = bool_argparse(args.predict) # Parse boolean predict argument

    if bool_training == True or bool_test == True:
        X_train, X_test, y_train, y_test = train_test_split_df(df) # Split data into train and test sets

        if bool_training == True:
            model.search_best_params(X_train, y_train,param_grid,args.n_iter,args.cv) # Search for best parameters
            model.train_final_model(X_train, y_train) # Train final model

        if bool_test == True:
            model.evaluate_model(X_test, y_test) # Evaluate model performance on test set

    if bool_predict == True:
        X = df.iloc[:, :-1]  # Extract features for prediction
        predicted = model.predict(X) # Make predictions

        df_tot["Predicted S"] = predicted + 1. # Add predicted values to DataFrame
        df_tot.to_csv('src/data_with_prediction.csv', index=False) # Save DataFrame with predictions


if __name__ == '__main__':

    ##############################################################################################################
    # Parsing command line arguments
    argparser = argparse.ArgumentParser()
    
    # RUNNING PARAMETERS
    argparser.add_argument("--dir", default='data', type=str, help="Name of the folder in which the data are stored")
    argparser.add_argument("--training", default="True", type=str, help="Perform the training and Hyperparameters Tuning or not")
    argparser.add_argument("--test", default="True", type=str, help="Testing the saved model")
    argparser.add_argument("--predict", default="False", type=str, help="Perform the prediction of the input data")

    # MODEL PARAMETERS
    argparser.add_argument("--model", default="xgb", type=str, help="Model to use")
    argparser.add_argument("--n_iter", default=15, type=int, help="Number of iterations for Hyperparameters Tuning")
    argparser.add_argument("--cv", default=3, type=int, help="Number of folds for Hyperparameters Tuning")

    #COMPUTE PARAMETERS
    argparser.add_argument("--seed", default=42, type=int, help="seed to use")

    args = argparser.parse_args() # Parse command line arguments

    np.random.seed(args.seed) # Set seed for reproducibility
    main(args) # Call main function with parsed arguments