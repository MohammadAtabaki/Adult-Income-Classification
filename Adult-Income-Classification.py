import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


from google.colab import files
files.upload()

!unzip archive.zip

df = pd.read_csv('/content/adult.csv')

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

temp_X = pd.concat([X_train, X_test], axis=0)
temp_X_new = pd.get_dummies(temp_X)

X_train = temp_X_new[:len(X_train)]
X_test = temp_X_new[len(X_train):]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred , pos_label='>50K')

svc =SVC()
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

acc_svc = accuracy_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc , pos_label='>50K')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn , pos_label='>50K')

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf , pos_label='>50K')

xgb = XGBClassifier()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train)
y_test_xgb = le.transform(y_test)
xgb.fit(X_train, y_train_xgb)
y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
f1_xgb = f1_score(y_test_xgb, y_pred_xgb )

metric_df = pd.DataFrame({'Model': ['Logistic Regression', 'SVC', 'KNN', 'Random Forest', 'XGBoost'],
                          'Accuracy': [acc_lr, acc_svc, acc_knn, acc_rf, acc_xgb],
                          'F1 Score': [f1_lr, f1_svc, f1_knn, f1_rf, f1_xgb]})
metric_df = metric_df.sort_values(by='F1 Score', ascending=False)

metric_df.plot(kind='bar', x='Model', y=['Accuracy', 'F1 Score'], figsize=(10,5))
