from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X, y = data.data, data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = ___

# 3. Initialize and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=___,
    max_depth=___,
    random_state=42
)
rf_model.___(X_train, y_train)

# 4. Predict on test data
y_pred = rf_model.___(X_test)

# 5. Evaluate accuracy
accuracy = ___(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")