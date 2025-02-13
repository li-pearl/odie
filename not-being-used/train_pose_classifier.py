import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load pose dataset
df = pd.read_csv("pose_data.csv")

# Extract features (keypoints) and labels
X = df.iloc[:, :-1].values  # Keypoints
y = df.iloc[:, -1].values   # Pose labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, "models/pose_classifier.pkl")
print("Pose classifier trained and saved as models/pose_classifier.pkl")
