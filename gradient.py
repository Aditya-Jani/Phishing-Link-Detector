import pandas as pd                                                           # data analysis mate
from sklearn.model_selection import train_test_split                          # split karse data
from sklearn.ensemble import GradientBoostingClassifier                       # predict karva mate 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # value distribution and assignment mate

data = pd.read_csv('url_specifications.csv')

X = data.iloc[:, 1:-1]                                                                                 
y = data.iloc[:, -1]                                                            
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     #Split data nu training and testing karse.

gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)    #parameters che 
gb_clf.fit(X_train, y_train)                                #glb data nae train karse.

predictions = gb_clf.predict(X_test)                        
print("Accuracy: ", accuracy_score(y_test, predictions))