import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

# ==========================================
# 0. SETUP
# ==========================================
output_folder = 'chart'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==========================================
# 1. DESIGN PHASE: Advanced Data Prep
# ==========================================
print("--- Loading Data... ---")
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# A. Standard Geometry (Distance & Angle)
df['x_m'] = df['X'] * 105
df['y_m'] = df['Y'] * 68
df['distance'] = np.sqrt((105 - df['x_m'])**2 + (34 - df['y_m'])**2)

def calc_angle(x, y):
    a = np.sqrt((105 - x)**2 + (30.34 - y)**2)
    b = np.sqrt((105 - x)**2 + (37.66 - y)**2)
    c = 7.32
    if a * b == 0: return 0
    return np.degrees(np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)))

df['angle'] = df.apply(lambda row: calc_angle(row['x_m'], row['y_m']), axis=1)
df['is_goal'] = df['result'].apply(lambda x: 1 if x == 'Goal' else 0)

# B. FILTER: Long Range Only
long_range_df = df[df['distance'] > 24].copy()
print(f"Analyzing {len(long_range_df)} Long-Range Shots")

# C. ADVANCED FEATURES (Proxies for Phase/Structure)
# We select categorical variables that describe the "Situation"
features_to_use = ['distance', 'angle', 'minute', 'situation', 'lastAction', 'shotType']

# Prepare the dataset
X = long_range_df[features_to_use]
y = long_range_df['is_goal']

# D. ONE-HOT ENCODING (Crucial Step)
# Computers can't read text like "OpenPlay". We convert them to numbers (0/1).
X = pd.get_dummies(X, columns=['situation', 'lastAction', 'shotType'], drop_first=True)

print(f"Total Input Features after Encoding: {X.shape[1]}")
# Example features created: 'situation_OpenPlay', 'lastAction_Throughball', 'shotType_Head'

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. DEVELOPMENT PHASE: Training
# ==========================================
print("\n--- Training Advanced XGBoost Model ---")

ratio = float(np.sum(y == 0)) / np.sum(y == 1)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=150,        # Increased trees for more features
    max_depth=5,             # Deeper trees to catch complex interactions
    learning_rate=0.05,      # Slower learning for better accuracy
    scale_pos_weight=ratio,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)
joblib.dump(model, 'xgboost_long_range_model.pkl')
print("Advanced Model Saved.")

# ==========================================
# 3. EVALUATION (Did adding factors help?)
# ==========================================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc_score = roc_auc_score(y_test, y_prob)
print(f"\n--- RESULTS ---")
print(f"New ROC AUC Score: {auc_score:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 4. VISUALIZATION (Updated Feature Importance)
# ==========================================
# Now we can see if 'OpenPlay' or 'Volley' matters more than 'Distance'

plt.figure(figsize=(12, 8))
# Get feature importance
importance = model.get_booster().get_score(importance_type='gain')
# Sort and take top 15
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
features, scores = zip(*sorted_importance)

plt.barh(range(len(features)), scores, color='teal', align='center')
plt.yticks(range(len(features)), features)
plt.gca().invert_yaxis() # Highest importance on top
plt.title('What Drives a Goal? (Including Contextual Factors)')
plt.xlabel('Relative Importance (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Chart1_Advanced_Feature_Importance.png'), dpi=300)
print("Saved Chart 1: Advanced Feature Importance")

print("\nDONE. The model now considers Phase of Play and Structure proxies.")