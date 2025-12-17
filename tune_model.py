import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
import joblib

print("--- Starting Hyperparameter Tuning ---")

# 1. Load & Prep Data
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")
df['x_m'] = df['X'] * 105
df['y_m'] = df['Y'] * 68
df['distance'] = np.sqrt((105 - df['x_m'])**2 + (34 - df['y_m'])**2)

# Angle Calc
def calc_angle(x, y):
    a = np.sqrt((105 - x)**2 + (30.34 - y)**2)
    b = np.sqrt((105 - x)**2 + (37.66 - y)**2)
    c = 7.32
    if a * b == 0: return 0
    return np.degrees(np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)))

df['angle'] = df.apply(lambda row: calc_angle(row['x_m'], row['y_m']), axis=1)
df['is_goal'] = df['result'].apply(lambda x: 1 if x == 'Goal' else 0)

# Filter Long Range
data = df[df['distance'] > 24].copy()

# Features & Encoding
features = ['distance', 'angle', 'minute', 'situation', 'lastAction', 'shotType']
X = pd.get_dummies(data[features], columns=['situation', 'lastAction', 'shotType'], drop_first=True)
y = data['is_goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define the "Search Space" (The range of dials to turn)
param_grid = {
    'n_estimators': [100, 200, 300, 500],        # Number of trees
    'max_depth': [3, 4, 5, 6, 8],                # How deep each tree goes
    'learning_rate': [0.01, 0.05, 0.1, 0.2],     # How fast it learns
    'subsample': [0.7, 0.8, 0.9, 1.0],           # % of data to use per tree
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],    # % of features to use per tree
    'gamma': [0, 0.1, 0.2, 0.5],                 # Minimum loss reduction
    'scale_pos_weight': [10, 20, 30, 40]         # Weight for "Goal" class (Crucial!)
}

# 3. Setup the Tuner
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,              # Try 50 different combinations
    scoring='roc_auc',      # Optimize for AUC specifically
    cv=3,                   # Cross-validation (3-fold)
    verbose=1,
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

# 4. Run the Search (This will take 2-5 minutes)
print("Tuning in progress... please wait...")
search.fit(X_train, y_train)

# 5. Results
best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
new_auc = roc_auc_score(y_test, y_prob)

print("\n--- TUNING COMPLETE ---")
print(f"Best AUC Score: {new_auc:.4f}")
print("Best Parameters Found:")
print(search.best_params_)

# Save the best model
joblib.dump(best_model, 'xgboost_tuned_best.pkl')
print("\nSaved best model as 'xgboost_tuned_best.pkl'")