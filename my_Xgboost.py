import pandas as pd                                                             # data analysis mate
from sklearn.model_selection import train_test_split                            # split karse data
from xgboost import XGBClassifier                                               # tasks mate
from sklearn.metrics import accuracy_score                                      # function lavse aapda model maa

data = pd.read_csv('url_specifications.csv')

X = data.iloc[:,1:8]
Y = data.iloc[:,8]

seed = 7                                
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)    #Model train thase

model = XGBClassifier()
model.fit(X_train, y_train)                                                         # Model Predict 

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]                                        #last time prediction

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))                                                      #Model Evaluate thase.