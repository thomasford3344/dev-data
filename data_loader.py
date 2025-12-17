# data_loader.py
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
