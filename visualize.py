# visualize.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_pairplot(X, y):
    df = X.copy()
    df['target'] = y
    sns.pairplot(df, hue='target')
    plt.show()

def plot_feature_histograms(X):
    X.hist(figsize=(10,8))
    plt.show()
