#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier                                               # tasks mate
from sklearn.model_selection import GridSearchCV
from ExtractingFeatures import urlFeatures

# In[3]:

df = pd.read_csv("url_specifications.csv")
df.head()

# In[4]:

X_train, X_test, y_train, y_test = train_test_split(df[["URL Length", "Non Standard Ports", "HTTPS", "Special Characters", "Numeric Characters", "Number of Redirects", "Shortening of URL"]], df.Phishing, test_size=0.1)

# In[5]:

# model = LogisticRegression()
model = XGBClassifier()


# In[6]:

parameters = {'C': [0.1, 1, 10, 100, 1000],  
              'penalty': ['l1', 'l2'],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

grid = GridSearchCV(model, parameters, refit = True, verbose = 3)

grid.fit(X_train, y_train)

print("Best parameters : ", grid.best_params_)
print("Best estimator : ", grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print("Grid search model score : ", grid.score(X_test, y_test))

# %%