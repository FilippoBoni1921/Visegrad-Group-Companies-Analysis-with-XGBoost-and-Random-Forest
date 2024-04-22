import os
import pandas as pd
import numpy as np
import arff

class DataProcessor:
    def __init__(self, folder='data'):
        """
        Initialize the DataProcessor object.

        Parameters:
        - folder (str): Name of the folder containing data files. Default is 'data'.
        """
        self.folder = folder

    def find_data_directory(self):
        """
        Find the data directory.

        Returns:
        - data_path (str): Path of the data directory.
        """
        # Check if "data" directory exists in the current directory
        current_data_path = os.path.join(os.getcwd(),self.folder)
        if os.path.exists(current_data_path) and os.path.isdir(current_data_path):
            return current_data_path # Return the current data path
        
        # Check parent directory for "data" directory
        parent_data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
        if os.path.exists(parent_data_path) and os.path.isdir(parent_data_path):
            return parent_data_path # Return the parent data path
        
        # If "data" directory is not found, raise an error
        raise FileNotFoundError(f"Could not find {self.folder} directory.")

    def create_df(self):
        """
        Create DataFrame from ARFF files.

        Returns:
        - df_ls (list of pandas.DataFrame): List of DataFrames created from ARFF files.
        """

        folder = self.find_data_directory() # Finding the data directory
        files = sorted(os.listdir(folder)) # Sorting files in the directory
        df_ls = [] # Initializing list to store DataFrames

        for f in files:
            file = folder + "/" + f # Creating file path
            dataset = arff.load(open(file, 'r')) # Loading ARFF file
            df = pd.DataFrame(dataset['data']) # Creating DataFrame
            df.insert(0, 'Quarter', f[:7]) # Inserting 'Quarter' column
            df_ls.append(df) # Appending DataFrame to the list

        return df_ls

    @staticmethod
    def convert_to_float(val):
        """
        Convert a value to float.

        Parameters:
        - val: Value to be converted.

        Returns:
        - float_val: Converted float value.
        """
        try:
            return float(val) # Convert the value to float
        except ValueError:
            return val  # Return the original value for non-convertible strings

    def check_unique_strings(self, df_ls):
        """
        Check if DataFrame contains only one unique string.

        Parameters:
        - df_ls (list of pandas.DataFrame): List of DataFrames.

        Raises:
        - ValueError: If DataFrame contains multiple unique strings.
        """
        for i in range(len(df_ls)):
            df = df_ls[i].iloc[:, 2:] # Extracting relevant columns from DataFrame

            unique_strings = df.stack().unique() # Finding unique values in the DataFrame
            # Filter out non-string values
            strings = [string for string in unique_strings if isinstance(string, str)]

            if len(strings) > 1: # Checking if more than one unique string exists
                raise ValueError("The DataFrame contains multiple unique strings, 'm' is not the only one")

    @staticmethod
    def check_specific_value(df_ls, specific_value='m'):
        """
        Check if a specific value exists as the only unique value in a DataFrame column.

        Parameters:
        - df_ls (list of pandas.DataFrame): List of DataFrames.
        - specific_value (str): Specific value to be checked. Default is 'm'.
        """
        for i in range(len(df_ls)):
            df = df_ls[i]
            cols = df.columns[2:-1]

            for col in cols:
                # Count the occurrences of each unique value in the column
                value_counts = df[col].value_counts()

                # Check if there is only one unique value and if it matches the specific value
                if len(value_counts) == 1 and value_counts.index[0] == specific_value:
                    print(f"In DataFrame {i}, the column '{col}' contains only the specific value '{specific_value}'.")

    @staticmethod
    def replace_m(df_ls):
        """
        Replace 'm' values in DataFrame with the mean of non-'m' values.

        Parameters:
        - df_ls (list of pandas.DataFrame): List of DataFrames.

        Returns:
        - df_ls (list of pandas.DataFrame): List of DataFrames with 'm' replaced.
        """
        for i in range(len(df_ls)):
            df = df_ls[i]
            countries = np.unique(df[0].values) # Get unique country values
            cols = df.columns[2:-1] # Get relevant columns excluding the first two and the last one

            for col in cols: # Iterate over columns
                for country in countries:

                    mask1 = df[col] != 'm' # Create mask to filter out 'm' values
                    mask2 = df[0] == country # Create mask to filter out rows corresponding to the current country

                    mask_1_2_arr = df[mask1 & mask2][col].values # Apply both masks and get the values
                    mask_1_arr = df[mask1][col].values # Apply only the first mask and get the values

                    if len(mask_1_2_arr) != 0: # Check if there are non-'m' values for the current country
                        df[col].replace('m', np.mean(mask_1_2_arr), inplace=True) # Replace 'm' with mean of non-'m' values
                    elif len(mask_1_2_arr) == 0 and len(mask_1_arr) != 0:  # Check if there are non-'m' values for other countries
                        df[col].replace('m', np.mean(mask_1_arr), inplace=True) # Replace 'm' with mean of non-'m' values

        return df_ls