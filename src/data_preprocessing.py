""" Processing Data """

import pandas as pd
import numpy as np

# ================================================================================================================ #

class DataProcessor():
    """
    Class defining objects that preprocess, normalize and split the data.
    """
    def __init__(self, data_file_path:str):
        self.data_file_path = data_file_path
        self.data = None

    def preprocess_data(self, 
                        y_column:str,
                        normalize:bool=True) -> pd.DataFrame:
        """
        Preprocess data.
        """
        self.data = pd.read_csv(self.data_file_path)
        columns_of_dtype_object = self.data.select_dtypes(include='object').columns
        
        # One-hot encoding columns of dtype object.
        for column in columns_of_dtype_object:
            insert_loc = self.data.columns.get_loc(column)
            self.data = pd.concat([self.data.iloc[:,:insert_loc], pd.get_dummies(self.data.loc[:, [column]], drop_first=True, dtype=float), self.data.iloc[:,insert_loc+1:]], axis=1)
            self.data = self.data.rename(columns={self.data.columns[insert_loc] : column})

        if normalize:
            self.data = (self.data - self.data.mean())/self.data.std()
            
        assert y_column in self.data.columns, f'{y_column} is not in the list of columns.'

        # Obtaining the features X and label y
        X = self.data.drop(columns=[y_column], axis=1).to_numpy()
        y = self.data[y_column].to_numpy()
        # Adding a bias term to the feature matrix.
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X, y
    
    def split_data_into_training_validation_testing(self, X:np.array, y:np.array, test_size:float=0.8, val_size:float=0.9, random_state:int=12):
        """
        Splits the data into training, validation and testing data.
        """
        n = len(X)
        split = int(test_size*n)

        X_non_test, X_test = X[:split], X[split:]
        y_non_test, y_test = y[:split], y[split:]

        np.random.seed(random_state)

        n_non_test = len(X_non_test)

        split = int(val_size*n_non_test)

        indices = np.random.permutation(n_non_test)
        train_indices = indices[:split]
        val_indices = indices[split:]

        X_train, X_val = X_non_test[train_indices], X_non_test[val_indices]
        y_train, y_val = y_non_test[train_indices], y_non_test[val_indices]

        return X_train, X_val, X_test, y_train, y_val, y_test

    
