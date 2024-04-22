
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from src.data_load_and_preproc import DataProcessor
from src.data_covid_analyzer import DataAnalyzer
from sklearn.preprocessing import LabelEncoder

folder = "data"
data_processor = DataProcessor(folder)

@pytest.fixture()
def create_df():
    """
    Fixture to create DataFrame.
    """
    df_ls = data_processor.create_df() # Create DataFrame ls
    return df_ls

@pytest.fixture()
def convert_to_float(create_df):
    """
    Fixture to convert values to float.
    """
    # Call create_df method
    df_ls = create_df
    for df in df_ls:
        df.iloc[:, 2:] = df.iloc[:, 2:].applymap(data_processor.convert_to_float) # Convert values to float
    return df_ls

@pytest.fixture()
def replace_m(convert_to_float):
    """
    Fixture to replace 'm' values.
    """
    return data_processor.replace_m(convert_to_float) # Replace 'm' values in the DataFrame list

@pytest.fixture()
def check_not_nan(replace_m):
    """
    Fixture to check if there are no NaN values.
    """
    with patch.object(DataAnalyzer, 'plot_variations') as mock_plot_variations: # Mock plot_variations method
        data_analyzer = DataAnalyzer(replace_m) 
        variations_all = data_analyzer.analyze_data() #Ger variations

        return variations_all

def df_prep_for_model(df_tot):
    """
    Prepare DataFrame for modeling.
    """
    rows_dropped = len(df_tot[df_tot[83] == 'm'])
    df_tot.drop(df_tot[df_tot[83] == 'm'].index, inplace=True)

    df_tot2 = df_tot.copy()
    df_tot2[83] = df_tot2[83]-1
    df_tot2[83] = df_tot2[83].astype(int)
    label_encoder = LabelEncoder()
    df_tot2[0] = label_encoder.fit_transform(df_tot2[0])

    df = df_tot2.iloc[:,1:]

    return df


def test_create_df(create_df):
    """
    Test DataFrame creation.
    """
    assert isinstance(create_df, list) # Check if create_df returns a list
    for df in create_df:
        assert isinstance(df, pd.DataFrame) # Check if each element in the list is a DataFrame

def test_convert_float(convert_to_float):
    """
    Test converting values to float.
    """
    for df in convert_to_float:
        arr = df.iloc[:, 2:].values # Select values to convert
        mask = arr != 'm' # Create a mask for non-'m' values
        masked_indices = np.where(mask) # Get indices of non-'m' values
        arr_masked = arr[masked_indices] # Select non-'m' values
        assert np.all(np.isreal(arr_masked)) # Check if all non-'m' values are real numbers

def test_replace_m(replace_m):
    """
    Test replacing 'm' values.
    """
    # Access the DataFrame returned by the replace_m fixture
    for df in replace_m:
    # Check if the 'm' values are replaced correctly
        assert 'm' not in df.iloc[:, 2:-1].values.flatten() # Check if 'm' values are not present in the DataFrame


def test_check_not_nan(check_not_nan):
    """
    Test for NaN values.
    """
    arr = np.asarray(check_not_nan) # Convert result to array
    arr = arr.flatten() 

    assert not np.isnan(arr).any() # Check if there are no NaN values


def test_df_prep_for_model(replace_m):
    """
    Test DataFrame preparation for modeling.
    """
    df_tot = pd.concat(replace_m)
    df = df_prep_for_model(df_tot)

    assert np.array_equal(np.unique(df[83].values), np.array([0, 1, 2, 3, 4, 5])) # Assert that the label column is decremented by 1 and converted to integer type

    assert np.array_equal(np.unique(df[0].values), np.array([0, 1, 2, 3, 4])) # Assert that the categorical column is encoded