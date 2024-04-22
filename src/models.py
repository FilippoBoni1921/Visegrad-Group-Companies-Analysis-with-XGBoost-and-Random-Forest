from sklearn.model_selection import RandomizedSearchCV, train_test_split
import sklearn.ensemble
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os


class XGBoostClassifier:
    def __init__(self, num_class=6, random_state=42):
        """
        Initialize the XGBoost classifier.

        Parameters:
        - num_class (int): Number of classes.
        - random_state (int): Random state for reproducibility.
        """
        self.num_class = num_class
        self.random_state = random_state
        self.best_params_ = None
        self.final_model = None

    def search_best_params(self, X_train, y_train, param_grid,n_iter,cv):
        """
        Search for the best hyperparameters using RandomizedSearchCV.

        Parameters:
        - X_train (array-like): Training features.
        - y_train (array-like): Training labels.
        - param_grid (dict): Parameter grid for RandomizedSearchCV.
        - n_iter (int): Number of iterations for RandomizedSearchCV.
        - cv (int): Number of cross-validation folds.

        Returns:
        - None
        """
        # Create a XGBoost classifier
        xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=self.num_class, random_state=self.random_state)

        # Create the RandomizedSearchCV object
        random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_grid,
                                           n_iter=n_iter, scoring='accuracy', cv=cv, verbose=3, random_state=self.random_state)

        # Perform RandomizedSearchCV on the training data
        random_search.fit(X_train, y_train)

        # Get the best parameters
        self.best_params_ = random_search.best_params_

    def train_final_model(self, X_train, y_train):
        """
        Train the final XGBoost model with the best parameters on the entire training set.

        Parameters:
        - X_train (array-like): Training features.
        - y_train (array-like): Training labels.

        Returns:
        - None
        """
        # Train the final model with the best parameters on the entire training set
        self.final_model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.num_class, random_state=self.random_state, **self.best_params_)
        self.final_model.fit(X_train, y_train)

        # Save the trained model
        with open('src/models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.final_model, f)
    
    def save_metrics(self,accuracy,recall,precision,f1,cm):
        """
        Save evaluation metrics and confusion matrix to files.

        Parameters:
        - accuracy (float): Test accuracy.
        - recall (float): Recall score.
        - precision (float): Precision score.
        - f1 (float): F1 score.
        - cm (array-like): Confusion matrix.

        Returns:
        - None
        """

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.title('XGB cf')
        plt.ylabel('Actual')
        # Save confusion matrix plot
        plt.savefig('src/plots/confusion_matrix_xgb.png')

        # Create a dictionary to store the scores
        scores = {
            "Test Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1
        }

        # Save the scores to a JSON file
        # Check if the file exists
        if os.path.isfile('src/test_results.json'):
            # Load existing data from the JSON file
            with open('src/test_results.json', 'r') as f:
                existing_data = json.load(f)
        else:
            # Create a new dictionary to store the scores
            existing_data = {}

        # Update the existing scores with the new scores
        existing_data["xgb_test_results"] = scores

        # Save the updated data back to the JSON file
        with open('src/test_results.json', 'w') as f:
            json.dump(existing_data, f, indent=4)


    def evaluate_model(self, X_test, y_test,file_path = 'src/models/xgboost_model.pkl'):
        """
        Evaluate the final XGBoost model on the test set.

        Parameters:
        - X_test (array-like): Test features.
        - y_test (array-like): Test labels.
        - file_path (str): Path to the saved model file.

        Returns:
        - None
        """
        # Load the saved model
        with open(file_path, 'rb') as f:
            self.final_model = pickle.load(f)

        # Evaluate the final model on the test set
        y_pred = self.final_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate recall, precision, and F1 score
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        #Save Metrics
        self.save_metrics(accuracy,recall,precision,f1,cm)

    def predict(self,X,file_path ='src/models/xgboost_model.pkl'):
        """
        Predict using the trained XGBoost model.

        Parameters:
        - X (array-like): Input features for prediction.
        - file_path (str): Path to the saved model file.

        Returns:
        - array-like: Predicted labels.
        """
        # Load the saved model
        with open(file_path, 'rb') as f:
            self.final_model = pickle.load(f)

        # Predict
        y_pred = self.final_model.predict(X)

        return y_pred

class RandomForestClassifier:
    def __init__(self, random_state=42):
        """
        Initialize the Random Forest classifier.

        Parameters:
        - random_state (int): Random state for reproducibility.
        """
        self.random_state = random_state
        self.best_params_ = None
        self.final_model = None

    def search_best_params(self, X_train, y_train, param_grid, n_iter, cv):
        """
        Search for the best hyperparameters using RandomizedSearchCV.

        Parameters:
        - X_train (array-like): Training features.
        - y_train (array-like): Training labels.
        - param_grid (dict): Parameter grid for RandomizedSearchCV.
        - n_iter (int): Number of iterations for RandomizedSearchCV.
        - cv (int): Number of cross-validation folds.

        Returns:
        - None
        """
        # Create a Random Forest classifier
        rf_clf = sklearn.ensemble.RandomForestClassifier(class_weight='balanced',random_state=self.random_state)

        # Create the RandomizedSearchCV object
        random_search = RandomizedSearchCV(estimator=rf_clf, param_distributions=param_grid,
                                           n_iter=n_iter, scoring='accuracy', cv=cv, verbose=3, random_state=self.random_state)

        # Perform RandomizedSearchCV on the training data
        random_search.fit(X_train, y_train)

        # Get the best parameters
        self.best_params_ = random_search.best_params_

    def train_final_model(self, X_train, y_train):
        """
        Train the final Random Forest model with the best parameters on the entire training set.

        Parameters:
        - X_train (array-like): Training features.
        - y_train (array-like): Training labels.

        Returns:
        - None
        """
        # Train the final model with the best parameters on the entire training set
        self.final_model = sklearn.ensemble.RandomForestClassifier(random_state=self.random_state, **self.best_params_)
        self.final_model.fit(X_train, y_train)

        # Save the trained model
        with open('src/models/randomforest_model.pkl', 'wb') as f:
            pickle.dump(self.final_model, f)
    
    def save_metrics(self,accuracy,recall,precision,f1,cm):
        """
        Save evaluation metrics and confusion matrix to files.

        Parameters:
        - accuracy (float): Test accuracy.
        - recall (float): Recall score.
        - precision (float): Precision score.
        - f1 (float): F1 score.
        - cm (array-like): Confusion matrix.

        Returns:
        - None
        """
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.title('RF cf')
        plt.ylabel('Actual')
        # Save confusion matrix plot
        plt.savefig('src/plots/confusion_matrix_rf.png')

        # Create a dictionary to store the scores
        scores = {
            "Test Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1
        }

        # Save the scores to a JSON file
        # Check if the file exists
        if os.path.isfile('src/test_results.json'):
            # Load existing data from the JSON file
            with open('src/test_results.json', 'r') as f:
                existing_data = json.load(f)
        else:
            # Create a new dictionary to store the scores
            existing_data = {}

        # Update the existing scores with the new scores
        existing_data["randomforest_test_results"] = scores

        # Save the updated data back to the JSON file
        with open('src/test_results.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

    def evaluate_model(self, X_test, y_test,file_path = 'src/models/randomforest_model.pkl'):
        """
        Evaluate the final Random Forest model on the test set.

        Parameters:
        - X_test (array-like): Test features.
        - y_test (array-like): Test labels.
        - file_path (str): Path to the saved model file.

        Returns:
        - None
        """
        # Load the saved model
        with open(file_path, 'rb') as f:
            self.final_model = pickle.load(f)

        # Evaluate the final model on the test set
        y_pred = self.final_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate recall, precision, and F1 score
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
 
        #Save Metrics
        self.save_metrics(accuracy,recall,precision,f1,cm)
    
    def predict(self,X,file_path='src/models/randomforest_model.pkl'):
        """
        Predict using the trained Random Forest model.

        Parameters:
        - X (array-like): Input features for prediction.
        - file_path (str): Path to the saved model file.

        Returns:
        - array-like: Predicted labels.
        """
        # Load the saved model
        with open(file_path, 'rb') as f:
            self.final_model = pickle.load(f)

        # Predict 
        y_pred = self.final_model.predict(X)

        return y_pred
    

