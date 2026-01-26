# main.py
from data_loader import load_data
from model import train_model, evaluate_model
from visualize import plot_pairplot, plot_feature_histograms

# Step 1: Load dataset
X_train, X_test, y_train, y_test = load_data()

# Step 2: Visualize dataset
plot_pairplot(X_train, y_train)
plot_feature_histograms(X_train)

# Step 3: Train model
model = train_model(X_train, y_train)

# Step 4: Evaluate model
evaluate_model(model, X_test, y_test)

# https://drive.google.com/file/d/1_93Rz8R0TBjlxHMtqeHTcjvDPlVAzXMQ/view?usp=sharing