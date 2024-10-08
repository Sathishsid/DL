import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='Target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print(f'ROC AUC: {roc_auc:.2f}')

sample_data = np.array([[14.5, 20.5, 95.0, 670.0, 0.1, 0.2, 0.3, 0.2, 0.2, 0.07, 0.3, 1.5, 2.0, 30.0, 0.005, 0.02, 0.02, 0.01, 0.015, 0.002, 16.5, 22.5, 110.0, 850.0, 0.12, 0.35, 0.4, 0.3, 0.3, 0.09]])
sample_data = scaler.transform(sample_data)
predicted_class = model.predict(sample_data)
predicted_prob = model.predict_proba(sample_data)

print(f'Predicted Class: {"Malignant" if predicted_class[0] == 0 else "Benign"}')
print(f'Probability of being Malignant: {predicted_prob[0][0]:.2f}')
print(f'Probability of being Benign: {predicted_prob[0][1]:.2f}')
