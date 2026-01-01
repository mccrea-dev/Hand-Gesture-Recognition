import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load CSV
df = pd.read_csv("features_top_activity.csv")

# Target labels
y = df["label"]

X = df.drop(columns=["label", "src", "t_start_ms", "t_end_ms", "src"])

print("Training shape:", X.shape)
print(y.value_counts())

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X.to_numpy(), y)

joblib.dump(model, "gesture_model.pkl")
print("Model saved.")
