from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_test_split_df(df, test_size=0.2, random_state=None):
    """
    Split the DataFrame into train and test sets.

    Parameters:
    - df (pandas.DataFrame): DataFrame to split.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int, RandomState instance or None): Controls the randomness of the training and testing indices.

    Returns:
    - tuple: Tuple containing X_train, X_test, y_train, y_test.
    """
    # Separate features and labels
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Labels (last column)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    return X_train, X_test, y_train, y_test
