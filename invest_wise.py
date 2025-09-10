# Capstone Project - Group 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Load the dataset
data = pd.read_csv('Mumbai_encoded.csv')

# Define features and targets (corrected targets list)
features = ['bhk', 'type_encoded', 'area', 'price', 'region_encoded', 'status_encoded', 'age_encoded']
targets = ['expected_roi(%)', 'demand_indicator', 'property_liquidity_index']  # Removed 'market_volatility_score'

# Select features and targets
X = data[features]
y = data[targets]

# Split for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor for multi-target prediction
rf_regressor = RandomForestRegressor(random_state=42)
multi_target_rf = MultiOutputRegressor(rf_regressor)
multi_target_rf.fit(X_train, y_train)

# Predict targets on test set
y_pred = multi_target_rf.predict(X_test)

# Evaluate and print metrics for Random Forest Regressor
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("Random Forest Regressor Evaluation:")
for i, target in enumerate(targets):
    print(f"{target} - MSE: {mse[i]:.4f}, R2 Score: {r2[i]:.4f}")

# Convert predictions to DataFrame for classification rule
y_pred_df = pd.DataFrame(y_pred, columns=targets)

# Classification rule for predicted targets
def classify_investment(row):
    if row['expected_roi(%)'] >= 9 and row['demand_indicator'] >= 7:
        return 'Good Investment'
    else:
        return 'Not so good Investment'

# Apply classification rule to predicted values
y_pred_df['investment_class'] = y_pred_df.apply(classify_investment, axis=1)

# Generate investment class labels from true target values for classifier training
def classify_on_targets(df):
    return df.apply(classify_investment, axis=1)

investment_train = classify_on_targets(y_train).reset_index(drop=True)
investment_test = classify_on_targets(y_test).reset_index(drop=True)

# Train Random Forest Classifier on features and true investment classes
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, investment_train)

# Predict investment class on test features
investment_pred = rf_classifier.predict(X_test)

# Print classification report for classifier evaluation
print("\nRandom Forest Classifier Evaluation:")
print(classification_report(investment_test, investment_pred))

import pickle

# Saving rf_classifier as .pk1 file
with open("invest_wise_classifier.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)
