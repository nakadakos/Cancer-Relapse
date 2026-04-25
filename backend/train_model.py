"""
train_model.py (v2 — Improved)
================================
Enhanced model training with:
- SMOTE oversampling for class imbalance
- Hyperparameter tuning via RandomizedSearchCV  
- Cross-validation for robust evaluation
- Feature importance extraction
- Comprehensive evaluation metrics
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    recall_score, precision_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'combined_cancer_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature definitions
NUMERIC_FEATURES = [
    'Age', 'BMI', 'Tumor_Stage', 'Tumor_Grade', 'Tumor_Size_cm',
    'Time_Since_Treatment_Months', 'Follow_Up_Visits'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Athleticity', 'Smoking_Alcohol_History', 'Cancer_Type',
    'Lymph_Nodes_Involved', 'Metastasis', 'Tumor_Type', 'Hormone_Receptor',
    'Gene_Mutations', 'Surgery_Type', 'Chemotherapy', 'Radiation_Therapy',
    'Hormone_Therapy', 'Immunotherapy', 'Previous_Reoccurrence'
]


def load_data():
    """Load and prepare the combined dataset."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    if 'Data_Source' in df.columns:
        df = df.drop(columns=['Data_Source'])

    X = df.drop(columns=['Relapse'])
    y = df['Relapse'].apply(lambda x: 1 if x == 'Yes' else 0)

    print(f"Dataset: {len(df)} records, {len(X.columns)} features")
    print(f"Class distribution: {y.value_counts().to_dict()} (1=Relapse, 0=No Relapse)")
    print(f"Relapse rate: {y.mean()*100:.1f}%")
    return X, y


def build_preprocessor():
    """Build the sklearn preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )


def train_and_evaluate():
    """Train models with SMOTE + hyperparameter tuning."""
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor()

    # --- Phase 1: Transform data and apply SMOTE ---
    print("\n--- Phase 1: Preprocessing + SMOTE ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"After SMOTE: {len(X_train_resampled)} samples (from {len(X_train)})")
    print(f"  Class 0: {(y_train_resampled == 0).sum()}, Class 1: {(y_train_resampled == 1).sum()}")

    # --- Phase 2: Hyperparameter Tuning ---
    print("\n--- Phase 2: Hyperparameter Tuning ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model configs with hyperparameter search spaces
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.3]
            }
        }
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0
    results = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION (with SMOTE + Tuning)")
    print("=" * 60)

    for name, config in model_configs.items():
        print(f"\n--- {name} ---")
        print("  Tuning hyperparameters...")

        search = RandomizedSearchCV(
            config['model'], config['params'],
            n_iter=30, cv=cv, scoring='accuracy',
            random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X_train_resampled, y_train_resampled)

        tuned_model = search.best_estimator_
        y_pred = tuned_model.predict(X_test_processed)

        try:
            y_prob = tuned_model.predict_proba(X_test_processed)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None
            roc_auc = 0.0

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Cross-val scores
        cv_scores = cross_val_score(tuned_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')

        print(f"  Best Params: {search.best_params_}")
        print(f"  Accuracy:  {acc:.4f}  (CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(classification_report(y_test, y_pred, target_names=['No Relapse', 'Relapse']))

        results[name] = {
            'accuracy': round(acc, 4),
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'cv_recall_mean': round(cv_scores.mean(), 4),
            'cv_recall_std': round(cv_scores.std(), 4),
            'confusion_matrix': cm.tolist(),
            'best_params': {k: str(v) for k, v in search.best_params_.items()}
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = tuned_model

    # --- Phase 3: Save best model as full pipeline ---
    print("\n" + "=" * 60)
    print(f"BEST MODEL (Optimized for Accuracy): {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("=" * 60)

    # Create a full pipeline (preprocessor + model) for serving
    # The preprocessor is already fitted from Phase 1; the best_model was trained on SMOTE data
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),  # already fitted on X_train
        ('classifier', best_model)       # trained on SMOTE-resampled data
    ])

    model_path = os.path.join(MODEL_DIR, 'model_pipeline.pkl')
    joblib.dump(full_pipeline, model_path)
    print(f"Saved model to: {model_path}")

    # Save evaluation results
    eval_results = {
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'all_results': results,
        'features': {
            'numeric': NUMERIC_FEATURES,
            'categorical': CATEGORICAL_FEATURES
        },
        'dataset_size': len(X),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'smote_applied': True,
        'hyperparameter_tuning': True
    }

    eval_path = os.path.join(MODEL_DIR, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to: {eval_path}")

    # Feature importance
    try:
        if hasattr(best_model, 'feature_importances_'):
            feature_names = (NUMERIC_FEATURES +
                           list(preprocessor.named_transformers_['cat']
                                .get_feature_names_out(CATEGORICAL_FEATURES)))
            importances = best_model.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

            print("\nTop 15 Feature Importances:")
            for feat, imp in feat_imp[:15]:
                print(f"  {feat}: {imp:.4f}")

            feat_imp_dict = {feat: round(float(imp), 4) for feat, imp in feat_imp[:20]}
            imp_path = os.path.join(MODEL_DIR, 'feature_importances.json')
            with open(imp_path, 'w') as f:
                json.dump(feat_imp_dict, f, indent=2)
        elif hasattr(best_model, 'coef_'):
            feature_names = (NUMERIC_FEATURES +
                           list(preprocessor.named_transformers_['cat']
                                .get_feature_names_out(CATEGORICAL_FEATURES)))
            importances = np.abs(best_model.coef_[0])
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

            print("\nTop 15 Feature Importances (absolute coefficients):")
            for feat, imp in feat_imp[:15]:
                print(f"  {feat}: {imp:.4f}")

            feat_imp_dict = {feat: round(float(imp), 4) for feat, imp in feat_imp[:20]}
            imp_path = os.path.join(MODEL_DIR, 'feature_importances.json')
            with open(imp_path, 'w') as f:
                json.dump(feat_imp_dict, f, indent=2)
    except Exception as e:
        print(f"Could not extract feature importances: {e}")


if __name__ == "__main__":
    train_and_evaluate()
