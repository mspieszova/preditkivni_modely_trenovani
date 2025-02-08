# Import libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Explore the Data
df=pd.read_csv("health_data.csv")
print(df.head)

# Check for missing values
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)
duplicates = df[df.duplicated()]
print(f"Počet duplicitních řádků: {duplicates.shape[0]}")
if not duplicates.empty:
    print(duplicates)

# Display basic statistics
print(df.describe())

# Step 2: Prepare the Data for Modeling
X = df[['Age', 'BMI', 'BloodPressure']]
Y = df['DiseaseStatus']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  

# Step 3: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
print(y_pred)




# Calculate and display evaluation metrics
accuracy = accuracy_score(Y_test, Y_train)
precision = precision_score(Y_test, Y_train)
recall = recall_score(Y_test, Y_train)
f1 = f1_score(Y_test, Y_train)
print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Step 5: Feature Importance and Interpretation


# Plot feature importances

