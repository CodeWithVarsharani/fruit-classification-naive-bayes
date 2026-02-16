# =====================================================
# FRUIT CLASSIFICATION USING MACHINE LEARNING
# =====================================================

# ---------------------------
# 1️⃣ Import Libraries
# ---------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------
# 2️⃣ Create Dataset
# ---------------------------

data = {
    "weight": [150,170,140,130,180,160,170,120,110,100,155,165,175],
    "sweetness": [7,6,8,7,5,4,6,9,8,9,6,5,4],
    "color": ["Red","Red","Red","Red",
              "Orange","Orange","Orange",
              "Yellow","Yellow","Yellow",
              "Red","Orange","Yellow"],
    "fruit": ["Apple","Apple","Apple","Apple",
              "Orange","Orange","Orange",
              "Banana","Banana","Banana",
              "Apple","Orange","Banana"]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# ---------------------------
# 3️⃣ Data Preprocessing
# ---------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode categorical columns
le_color = LabelEncoder()
le_fruit = LabelEncoder()

df["color"] = le_color.fit_transform(df["color"])
df["fruit"] = le_fruit.fit_transform(df["fruit"])

# Features and Target
X = df[["weight", "sweetness", "color"]]
y = df["fruit"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ---------------------------
# 4️⃣ Data Visualization
# ---------------------------

plt.figure()
sns.scatterplot(x=df["weight"], y=df["sweetness"], hue=df["fruit"])
plt.title("Weight vs Sweetness")
plt.show()

plt.figure()
sns.heatmap(pd.DataFrame(X).corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# ---------------------------
# 5️⃣ Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------
# 6️⃣ Train Models
# ---------------------------

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))


# ---------------------------
# 7️⃣ Model Evaluation
# ---------------------------

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.show()


# ---------------------------
# 8️⃣ Compare Model Accuracies
# ---------------------------

accuracies = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Decision Tree": accuracy_score(y_test, y_pred_dt),
    "Random Forest": accuracy_score(y_test, y_pred_rf)
}

plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.show()


# ---------------------------
# 9️⃣ Test with New Sample
# ---------------------------

# Example: weight=140, sweetness=8, color=Yellow
new_sample = [[140, 8, le_color.transform(["Yellow"])[0]]]
new_sample = scaler.transform(new_sample)

prediction = rf.predict(new_sample)

print("\nPredicted Fruit:",
      le_fruit.inverse_transform(prediction)[0])
