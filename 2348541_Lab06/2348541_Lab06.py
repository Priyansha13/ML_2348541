#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

# for label encoding
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
df = pd.read_csv("insurance.csv")

# Check the dataset's structure
df.head()


# In[3]:


#checking for null values
df.isnull().sum()


# - There are no null values

# ### Encoding

# In[6]:


# Enabling Label encoder
Label_encoder = preprocessing.LabelEncoder()


# In[7]:


# Encoding qualitative variables to quantitave for further analysis
df['sex']=Label_encoder.fit_transform(df['sex'])
df['region']=Label_encoder.fit_transform(df['region'])
df['smoker']=Label_encoder.fit_transform(df['smoker'])


# ### Linear Regression

# In[9]:


# Define features (X) and target variable (y)
X = df.drop('charges', axis=1)
y = df['charges']

# Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[11]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[12]:


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
rss = np.sum(np.square(y_test - y_pred))
explained_var = explained_variance_score(y_test, y_pred)

# Calculate adjusted R-squared
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Display the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Adjusted R-squared Score: {adj_r2}")
print(f"Residual Sum of Squares (RSS): {rss}")
print(f"Explained Variance Score: {explained_var}")


# In[13]:


# Get feature coefficients
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)


# In[14]:


# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o', linestyle='', alpha=0.7)
plt.plot(y_pred, label='Predicted', marker='o', linestyle='', alpha=0.7)
plt.xlabel("Data Points")
plt.ylabel("Insurance Charges")
plt.title("Actual vs. Predicted Insurance Charges")
plt.legend()
plt.show()


# In[16]:


# Calculating residuals
residuals = y_test - y_pred

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("Predicted Insurance Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[17]:


# Create a distribution plot of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution Plot")
plt.show()


# In[18]:


# Example for age feature
plt.figure(figsize=(10, 6))
plt.scatter(X_test['age'], residuals)
plt.xlabel("Age")
plt.ylabel("Residuals")
plt.title("Residuals vs. Age")
plt.show()


# In[19]:



from scipy.stats import probplot

plt.figure(figsize=(10, 6))
probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()


# In[ ]:




