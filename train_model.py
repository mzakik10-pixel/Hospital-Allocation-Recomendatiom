"""
CrowdAID - ML Model Training Script
Train Random Forest dan Decision Tree models untuk hospital classification

Usage:
    python train_model.py
    
Output:
    - model_random_forest.pkl
    - model_decision_tree.pkl
    - label_encoders.pkl
    - model_metadata.json
    - feature_columns.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

print("="*70)
print("CrowdAID - ML MODEL TRAINING")
print("="*70)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/6] Loading data...")
df_hospital = pd.read_csv('Hospital_Banten.csv', sep=';')
print(f"✅ Loaded {len(df_hospital)} hospitals")

# ============================================
# 2. CREATE TRAINING DATA
# ============================================
print("\n[2/6] Creating training data...")

data_records = []

for idx, row in df_hospital.iterrows():
    hospital_type = row['jenis']
    hospital_class = row['kelas']
    capacity = row['total_tempat_tidur']
    services = row['total_layanan']
    staff = row['total_tenaga_kerja']
    
    # Generate labels based on rules
    conditions_rules = {
        'Gejala Ringan': 0,  # Always not suitable
        'Penyakit Dalam': 1 if (hospital_class == 'C' and 'Umum' in hospital_type) else 0,
        'Bedah': 1 if (hospital_class == 'C' and ('Bedah' in hospital_type or 'Umum' in hospital_type)) else 0,
        'Anak': 1 if (hospital_class == 'C' and ('Ibu dan Anak' in hospital_type or 'Umum' in hospital_type)) else 0,
        'Kebidanan': 1 if (hospital_class == 'C' and ('Ibu dan Anak' in hospital_type or 'Umum' in hospital_type)) else 0,
        'Gigi': 1 if hospital_class == 'D' else 0,
        'Banyak Spesialis': 1 if hospital_class == 'B' else 0,
    }
    
    for condition, is_suitable in conditions_rules.items():
        data_records.append({
            'hospital_type': hospital_type,
            'hospital_class': hospital_class,
            'capacity': capacity,
            'services': services,
            'staff': staff,
            'condition': condition,
            'is_suitable': is_suitable
        })

df_train = pd.DataFrame(data_records)
print(f"✅ Created {len(df_train)} training samples")

# ============================================
# 3. ENCODE FEATURES
# ============================================
print("\n[3/6] Encoding features...")

le_type = LabelEncoder()
le_class = LabelEncoder()
le_condition = LabelEncoder()

df_train['hospital_type_encoded'] = le_type.fit_transform(df_train['hospital_type'])
df_train['hospital_class_encoded'] = le_class.fit_transform(df_train['hospital_class'])
df_train['condition_encoded'] = le_condition.fit_transform(df_train['condition'])

print(f"✅ Encoded {len(le_type.classes_)} hospital types")
print(f"✅ Encoded {len(le_class.classes_)} hospital classes")
print(f"✅ Encoded {len(le_condition.classes_)} conditions")

# Prepare features and target
feature_columns = ['hospital_type_encoded', 'hospital_class_encoded', 
                   'capacity', 'services', 'staff', 'condition_encoded']
X = df_train[feature_columns]
y = df_train['is_suitable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training: {len(X_train)} samples, Test: {len(X_test)} samples")

# ============================================
# 4. TRAIN MODELS
# ============================================
print("\n[4/6] Training models...")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained")

# Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)
dt_model.fit(X_train, y_train)
print("✅ Decision Tree trained")

# ============================================
# 5. EVALUATE MODELS
# ============================================
print("\n[5/6] Evaluating models...")

# Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\n📊 Random Forest Accuracy: {accuracy_rf*100:.2f}%")

# Decision Tree
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"📊 Decision Tree Accuracy: {accuracy_dt*100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Top 3 Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']*100:.1f}%")

# ============================================
# 6. SAVE MODELS
# ============================================
print("\n[6/6] Saving models...")

# Save Random Forest
with open('model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Saved: model_random_forest.pkl")

# Save Decision Tree
with open('model_decision_tree.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("✅ Saved: model_decision_tree.pkl")

# Save encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'hospital_type': le_type,
        'hospital_class': le_class,
        'condition': le_condition
    }, f)
print("✅ Saved: label_encoders.pkl")

# Save feature columns
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
print("✅ Saved: feature_columns.json")

# Save metadata
metadata = {
    'model_type': 'Random Forest Classifier',
    'n_estimators': 100,
    'max_depth': 10,
    'accuracy_train': float(accuracy_rf),
    'accuracy_test': float(accuracy_rf),
    'feature_importance': feature_importance.to_dict('records'),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'feature_columns': feature_columns,
    'conditions': le_condition.classes_.tolist(),
    'hospital_types': le_type.classes_.tolist(),
    'hospital_classes': le_class.classes_.tolist()
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: model_metadata.json")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n🎯 Best Model: Random Forest")
print(f"📊 Accuracy: {accuracy_rf*100:.2f}%")
print(f"📁 Files saved: 5 files")
print(f"\n🚀 Ready to use with ml_predictor.py!")
print("="*70)
