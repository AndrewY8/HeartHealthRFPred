import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score





# cleaned dataset
df = pd.read_csv("framingham.csv")
df.drop(['education'], inplace = True, axis = 1)
df_cleaned = df.dropna()

#selected features
features = ['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
X = np.asarray(df_cleaned[features])
y = np.asarray(df_cleaned['TenYearCHD'])

#30 / 70 test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=8)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

#Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=df_cleaned, palette = "colorblind")
plt.show()