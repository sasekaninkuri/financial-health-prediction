# ============================================================
# AGGRESSIVE PIPELINE - Target 93% Macro F1
# ============================================================
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
test  = pd.read_csv('/Users/damacm1147/Downloads/Test_1.csv')
train = pd.read_csv('/Users/damacm1147/Downloads/Train_1.csv')
print(f"Train shape: {train.shape} | Test shape: {test.shape}")
print(f"\nTarget:\n{train['Target'].value_counts()}")
print(f"\nHigh per country:\n{train[train['Target']=='High']['country'].value_counts()}")

# ============================================================
# STEP 2: PREPROCESSING
# ============================================================
for df in [train, test]:
    df['country'] = df['country'].str.strip().str.title()

test_ids   = test['ID'].copy()
y_raw      = train['Target'].copy()
train_feat = train.drop(columns=['ID', 'Target']).copy()
test_feat  = test.drop(columns=['ID']).copy()

# ============================================================
# STEP 3: ACCESS FEATURES (before encoding while still strings)
# ============================================================
def add_access_features(df):
    df = df.copy()
    df['financial_access_score'] = (
        (df['has_credit_card']      == 'Have now').astype(int) +
        (df['has_debit_card']       == 'Have now').astype(int) +
        (df['has_internet_banking'] == 'Have now').astype(int) +
        (df['has_loan_account']     == 'Have now').astype(int) +
        (df['has_mobile_money']     == 'Have now').astype(int)
    )
    df['insurance_score'] = (
        (df['has_insurance']           == 'Have now').astype(int) +
        (df['medical_insurance']       == 'Have now').astype(int) +
        (df['funeral_insurance']       == 'Have now').astype(int) +
        (df['motor_vehicle_insurance'] == 'Have now').astype(int)
    )
    df['total_financial_score'] = df['financial_access_score'] + df['insurance_score']
    df['has_any_insurance']     = (df['insurance_score'] > 0).astype(int)
    df['has_any_banking']       = (df['financial_access_score'] > 0).astype(int)
    df['positive_attitude']     = (
        (df['attitude_stable_business_environment'] == 'Yes') &
        (df['attitude_more_successful_next_year']   == 'Yes')
    ).astype(int)
    df['worried_business']      = (df['attitude_worried_shutdown'] == 'Yes').astype(int)
    df['uses_informal_finance'] = (
        (df['uses_informal_lender']       == 'Have now') |
        (df['uses_friends_family_savings']== 'Have now')
    ).astype(int)
    df['keeps_records_flag']    = (df['keeps_financial_records'] == 'Yes').astype(int)
    df['offers_credit_flag']    = (df['offers_credit_to_customers'] == 'Yes').astype(int)
    return df

train_feat = add_access_features(train_feat)
test_feat  = add_access_features(test_feat)

print(f"\nfinancial_access_score dist:\n{train_feat['financial_access_score'].value_counts()}")
print(f"insurance_score dist:\n{train_feat['insurance_score'].value_counts()}")

# ============================================================
# STEP 4: NUMERIC FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    df = df.copy()
    df['profit_proxy']       = df['business_turnover'] - df['business_expenses']
    df['expense_ratio']      = df['business_expenses']  / (df['business_turnover'] + 1)
    df['income_vs_expense']  = df['personal_income']    / (df['business_expenses'] + 1)
    df['turnover_per_age']   = df['business_turnover']  / (df['owner_age'] + 1)
    df['income_per_age']     = df['personal_income']    / (df['owner_age'] + 1)
    df['turnover_vs_income'] = df['business_turnover']  / (df['personal_income'] + 1)
    df['turnover_per_year']  = df['business_turnover']  / (df['business_age_years'] + 1)
    df['total_age_months']   = (df['business_age_years'].fillna(0) * 12 +
                                df['business_age_months'].fillna(0))
    df['is_new_business']    = (df['business_age_years'] <= 1).astype(int)
    df['is_mature_business'] = (df['business_age_years'] >= 5).astype(int)
    for col in ['personal_income','business_expenses','business_turnover','profit_proxy']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    for col in ['personal_income','business_turnover']:
        df[f'sqrt_{col}'] = np.sqrt(df[col].clip(lower=0))
    return df

train_feat = engineer_features(train_feat)
test_feat  = engineer_features(test_feat)

# ============================================================
# STEP 5: IMPUTATION
# ============================================================
num_cols = train_feat.select_dtypes(include='number').columns.tolist()
cat_cols = [c for c in train_feat.select_dtypes(include='object').columns if c != 'country']

for col in num_cols:
    train_feat[col] = train_feat.groupby(train_feat['country'])[col]\
                        .transform(lambda x: x.fillna(x.median()))
    train_feat[col] = train_feat[col].fillna(train_feat[col].median())
    test_feat[col]  = test_feat.groupby(test_feat['country'])[col]\
                        .transform(lambda x: x.fillna(x.median()))
    test_feat[col]  = test_feat[col].fillna(train_feat[col].median())

for col in cat_cols:
    for df in [train_feat, test_feat]:
        df[col] = df.groupby('country')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))
        df[col] = df[col].fillna('Unknown')

print(f"\nMissing - Train: {train_feat.isnull().sum().sum()} | Test: {test_feat.isnull().sum().sum()}")

# ============================================================
# STEP 6: ENCODE
# ============================================================
all_cat_cols = train_feat.select_dtypes(include='object').columns.tolist()
combined     = pd.concat([train_feat, test_feat], axis=0).reset_index(drop=True)

for col in all_cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

train_enc = combined.iloc[:len(train_feat)].reset_index(drop=True)
test_enc  = combined.iloc[len(train_feat):].reset_index(drop=True)

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw.astype(str))
print(f"\nClasses: {le_target.classes_}")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

X      = train_enc.values.astype(float)
X_test = test_enc.values.astype(float)
feature_cols = list(train_enc.columns)

# ============================================================
# STEP 7: MODELS
# ============================================================
rf = RandomForestClassifier(
    n_estimators=1000, min_samples_leaf=1, min_samples_split=2,
    max_features='sqrt', class_weight='balanced_subsample',
    random_state=42, n_jobs=-1
)
gb = GradientBoostingClassifier(
    n_estimators=600, learning_rate=0.03, max_depth=6,
    min_samples_leaf=3, subsample=0.8, max_features='sqrt',
    random_state=42
)
et = ExtraTreesClassifier(
    n_estimators=1000, min_samples_leaf=1, min_samples_split=2,
    max_features='sqrt', class_weight='balanced_subsample',
    random_state=42, n_jobs=-1
)

# ============================================================
# STEP 8: CV — SMOTE INSIDE FOLDS
# ============================================================
print("\n" + "="*50)
print("CROSS-VALIDATION — SMOTE inside folds")
print("="*50)

skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42, k_neighbors=5)
rf_scores, gb_scores, et_scores, ens_scores = [], [], [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    rf.fit(X_tr_sm, y_tr_sm)
    gb.fit(X_tr_sm, y_tr_sm)
    et.fit(X_tr_sm, y_tr_sm)

    p_rf  = rf.predict_proba(X_val)
    p_gb  = gb.predict_proba(X_val)
    p_et  = et.predict_proba(X_val)
    p_ens = p_rf * 0.35 + p_gb * 0.40 + p_et * 0.25

    rf_scores.append(f1_score(y_val,  np.argmax(p_rf,  axis=1), average='macro'))
    gb_scores.append(f1_score(y_val,  np.argmax(p_gb,  axis=1), average='macro'))
    et_scores.append(f1_score(y_val,  np.argmax(p_et,  axis=1), average='macro'))
    ens_scores.append(f1_score(y_val, np.argmax(p_ens, axis=1), average='macro'))

    print(f"Fold {fold+1} | RF: {rf_scores[-1]:.4f} | GB: {gb_scores[-1]:.4f} | ET: {et_scores[-1]:.4f} | Ens: {ens_scores[-1]:.4f}")

print(f"\nRF  Mean F1 : {np.mean(rf_scores):.4f}")
print(f"GB  Mean F1 : {np.mean(gb_scores):.4f}")
print(f"ET  Mean F1 : {np.mean(et_scores):.4f}")
print(f"Ensemble F1 : {np.mean(ens_scores):.4f}")

# ============================================================
# STEP 9: FINAL TRAIN
# ============================================================
print("\nTraining final models on full data...")
X_sm, y_sm = smote.fit_resample(X, y)
rf.fit(X_sm, y_sm)
gb.fit(X_sm, y_sm)
et.fit(X_sm, y_sm)

p_train     = rf.predict_proba(X)*0.35 + gb.predict_proba(X)*0.40 + et.predict_proba(X)*0.25
train_preds = np.argmax(p_train, axis=1)

print("\nClassification Report:")
print(classification_report(y, train_preds, target_names=le_target.classes_))

cm = confusion_matrix(y, train_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Top 20 Feature Importances', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# ============================================================
# STEP 11: PREDICT + SAVE
# ============================================================
p_final    = rf.predict_proba(X_test)*0.35 + gb.predict_proba(X_test)*0.40 + et.predict_proba(X_test)*0.25
test_preds = le_target.inverse_transform(np.argmax(p_final, axis=1))

submission = pd.DataFrame({'ID': test_ids, 'Target': test_preds})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*50)
print("✅ submission.csv saved!")
print("="*50)
print(submission['Target'].value_counts())
print(submission.head())
