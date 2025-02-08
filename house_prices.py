# Import libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt



# Step 1: Load the data
df = pd.read_csv(r"house_prices_data.csv")
print(df.head())

# Step 2: Data exploration
X = df['Size']
Y = df['Price']
correlation = X.corr(Y)
print(f"Pearsonova korelace mezi velikostí a cenou: {correlation}")

# Step 3: Prepare the data for statsmodels
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.params)  


# Step 4: Perform linear regression
prediction = results.predict(X)

# Step 5: Visualize the regression line
plt.figure(figsize=(8, 5))
plt.scatter(df['Size'], df['Price'], color='blue', label='Data Points')  # Opravená osa X a Y
plt.plot(df['Size'], prediction, color='red', label='Prediction Line')
plt.legend()  # Opravená syntaxe
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Linear Regression: House Prices")
plt.show()
# Step 6: Save regression line

