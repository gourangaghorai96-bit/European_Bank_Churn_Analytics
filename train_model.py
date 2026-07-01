# ============================================
# train_model.py
# WHY: Retrain model locally
#      so scikit-learn versions match
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Step 1: Loading data...")

# WHY: Read Excel file (our dataset is .xlsx)
df = pd.read_csv('customer_churn.csv')
print(f"✅ Data Loaded! Shape: {df.shape}")

print("Step 2: Cleaning data...")

# WHY: Remove useless columns
df = df.drop(['Year', 'CustomerId', 'Surname'],
              axis=1, errors='ignore')

print("Step 3: Encoding...")

# WHY: Convert text to numbers
df_ml = pd.get_dummies(df,
                        columns=['Geography', 'Gender'],
                        drop_first=True)

print("Step 4: Preparing features...")

# WHY: Separate input and target
X = df_ml.drop(['Exited'], axis=1, errors='ignore')
y = df_ml['Exited']
print(f"✅ Features: {X.columns.tolist()}")

print("Step 5: Splitting data...")

# WHY: 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print("Step 6: Scaling data...")

# WHY: Bring all features to same range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Step 7: Training model...")

# WHY: Gradient Boosting gave best 87% accuracy
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

print("Step 8: Testing model...")

# WHY: Check accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

print("Step 9: Saving model...")

# WHY: Save new model with correct version
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("=" * 40)
print("✅ Model Saved Successfully!")
print("✅ Now run streamlit!")
print("=" * 40)