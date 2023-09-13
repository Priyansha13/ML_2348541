#!/usr/bin/env python
# coding: utf-8

# ### Importing required modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    log_loss,
)


# ### Loading dataset

# In[2]:


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[3]:


print("Dataset size")
print("Rows {} Columns {}".format(df.shape[0], df.shape[1]))


# In[4]:


print("Columns and data types")
pd.DataFrame(df.dtypes).rename(columns = {0:'dtype'})


# In[5]:


# checking for null values
df.isnull().any()


# In[6]:


# dropping irrelevant column
df = df.drop(columns='customerID', axis=1)
df.head()


# In[7]:


# describing target variable
df['Churn'].value_counts()


# - We can see that the target variable is imbalanced

# ### Encoding

# In[8]:


from sklearn.preprocessing import LabelEncoder
for column in df:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[9]:


df.info()


# In[10]:


df.head()


# ### Logistic Regression

# In[11]:


X = df.drop('Churn', axis=1)
y = df['Churn']


# In[12]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[14]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)
print(y_pred)


# ### Model Evaluation and Visualization:

# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[17]:


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
logloss = log_loss(y_test, model.predict_proba(X_test))

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Cohen's Kappa Score: {kappa:.2f}")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")
print(f"Log Loss: {logloss:.2f}")


# In[19]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[20]:


# Confusion Matrix
plt.figure(figsize=(6, 6))
plt.title('Confusion Matrix')
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['Not Churn', 'Churn'])
plt.yticks([0, 1], ['Not Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')
plt.show()


# In[21]:


# ROC AUC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[22]:


y_probs = model.predict_proba(X_test)[:, 1]


# In[23]:


# Visualize the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='blue', where='post', alpha=0.8)
plt.fill_between(recall, precision, alpha=0.2, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()


# In[24]:


# Sigmoid Curve Visualization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-6, 6, 100)
y = sigmoid(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, color='blue', lw=2)
plt.title('Sigmoid (Logistic) Curve')
plt.xlabel('Input Values (x)')
plt.ylabel('Sigmoid Output (y)')
plt.grid(True)
plt.show()


# In[25]:


from sklearn.model_selection import GridSearchCV


# In[26]:


# Hyperparameter Tuning
# Perform hyperparameter tuning to optimize the logistic regression model
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Tuning regularization strength (C)
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)


# In[27]:


# Get the best hyperparameter values
best_C = grid_search.best_params_['C']


# In[28]:


# Build a logistic regression model with the best hyperparameters
best_logistic_model = LogisticRegression(C=best_C, random_state=42)
best_logistic_model.fit(X_train, y_train)


# In[29]:


# Evaluate the model's performance with optimized hyperparameters
y_pred_best = best_logistic_model.predict(X_test)
roc_auc_best = roc_auc_score(y_test, best_logistic_model.predict_proba(X_test)[:, 1])

print("Optimized Model Metrics:")
print(f"Best C Value: {best_C}")
print(f"Optimized ROC AUC: {roc_auc_best:.2f}")


# In[30]:


# Define evaluation metrics for the baseline and tuned models
metrics_baseline = {
    'Accuracy': accuracy,
    'ROC AUC (Baseline)': roc_auc,
}

metrics_tuned = {
    'ROC AUC (Tuned)': roc_auc_best,
}

# Create bar plots to compare metrics
def plot_metrics_comparison(metrics1, metrics2, title):
    plt.figure(figsize=(10, 6))
    metrics_names = list(metrics1.keys())
    values1 = list(metrics1.values())
    values2 = list(metrics2.values())

    x = range(len(metrics_names))

    plt.bar(x, values1, width=0.4, label='Baseline', align='center', alpha=0.7)
    plt.bar([i + 0.4 for i in x], values2, width=0.4, label='Tuned', align='center', alpha=0.7)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks([i + 0.2 for i in x], metrics_names)
    plt.legend(loc='best')
    plt.ylim(0, 1.2 * max(max(values1), max(values2)))

    plt.show()

# Plot ROC AUC comparison
plot_metrics_comparison(metrics_baseline, metrics_tuned, 'ROC AUC Comparison (Before vs. After Tuning)')


# In[ ]:




