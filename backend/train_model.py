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
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
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
    """Build the sklearn preprocessing pipeline.

    Uses sub-pipelines so that:
    - Numeric NaNs are median-imputed before scaling.
    - Categorical NaNs are mode-imputed before OHE, preventing the encoder
      from creating spurious '_nan' feature columns that add pure noise.
    - drop='if_binary' eliminates perfect collinearity for Yes/No fields
      (e.g. Lymph_Nodes_Involved_Yes and _No are redundant; keeping both
      wastes feature space and inflates the importance of binary features).
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop='if_binary',   # removes _No duplicate for binary Yes/No columns
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
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

    # --- Phase 1: Fit preprocessor on raw training data; transform test set ---
    # SMOTE is applied *inside* the ImbPipeline during cross-validation so that
    # synthetic samples are never seen by the validation fold (no leakage).
    print("\n--- Phase 1: Preprocessing ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # --- Phase 2: Hyperparameter Tuning ---
    # Each model is wrapped in an ImbPipeline so SMOTE is applied only to the
    # training fold during cross-validation — preventing data leakage.
    print("\n--- Phase 2: Hyperparameter Tuning (scoring=recall) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # n_jobs: default 2 to avoid exhausting CPUs when run inside a server process.
    # Override by setting the TRAIN_N_JOBS environment variable (e.g. TRAIN_N_JOBS=4).
    n_jobs = int(os.environ.get('TRAIN_N_JOBS', 2))

    smote = SMOTE(random_state=42, sampling_strategy='auto')

    # Model configs with hyperparameter search spaces.
    # Param keys are prefixed with 'classifier__' because they sit inside ImbPipeline.
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, random_state=42),
            'params': {
                'classifier__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200, 300, 500],
                'classifier__max_depth': [5, 10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
            'params': {
                'classifier__n_estimators': [100, 200, 300, 500],
                'classifier__max_depth': [3, 5, 7, 9],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'classifier__gamma': [0, 0.1, 0.3]
            }
        }
    }

    best_model_name = None
    best_model = None
    best_recall = 0  # FIX: select by recall, not accuracy
    results = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION (with SMOTE per-fold + Tuning)")
    print("=" * 60)

    for name, config in model_configs.items():
        print(f"\n--- {name} ---")
        print("  Tuning hyperparameters...")

        # Wrap classifier + SMOTE into a single pipeline so SMOTE only sees
        # each CV training fold, never the validation fold.
        imb_pipe = ImbPipeline([
            ('smote', smote),
            ('classifier', config['model'])
        ])

        search = RandomizedSearchCV(
            imb_pipe, config['params'],
            n_iter=30, cv=cv, scoring='recall',
            random_state=42, n_jobs=n_jobs, verbose=0
        )
        search.fit(X_train_processed, y_train)

        tuned_pipe = search.best_estimator_  # ImbPipeline(smote + classifier)
        y_pred = tuned_pipe.predict(X_test_processed)

        try:
            y_prob = tuned_pipe.predict_proba(X_test_processed)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None
            roc_auc = 0.0

        acc       = accuracy_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        cm        = confusion_matrix(y_test, y_pred)

        # FIX: cross_val_score now uses recall (matches what was saved as cv_recall_*)
        cv_recall = cross_val_score(
            tuned_pipe, X_train_processed, y_train, cv=cv, scoring='recall'
        )

        print(f"  Best Params: {search.best_params_}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Recall:    {recall:.4f}  (CV recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f})")
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
            # FIX: key names now correctly describe recall CV scores
            'cv_recall_mean': round(cv_recall.mean(), 4),
            'cv_recall_std': round(cv_recall.std(), 4),
            'confusion_matrix': cm.tolist(),
            'best_params': {k: str(v) for k, v in search.best_params_.items()}
        }

        # FIX: select best model by recall, not accuracy
        if recall > best_recall:
            best_recall = recall
            best_model_name = name
            best_model = tuned_pipe

    # --- Phase 3: Save best model as full pipeline ---
    print("\n" + "=" * 60)
    print(f"BEST MODEL (Optimized for Recall): {best_model_name}")
    print(f"Recall: {best_recall:.4f}")
    print("=" * 60)

    # Compose the serving pipeline:
    # preprocessor (sklearn) → best ImbPipeline (smote + classifier)
    # At inference time SMOTE is skipped automatically because predict() is called,
    # not fit_resample(). The serving pipeline only uses the trained classifier.
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),   # fitted on X_train
        ('model', best_model)             # ImbPipeline(smote + classifier)
    ])

    model_path = os.path.join(MODEL_DIR, 'model_pipeline.pkl')
    joblib.dump(full_pipeline, model_path)
    print(f"Saved model to: {model_path}")

    # Save evaluation results
    eval_results = {
        'best_model': best_model_name,
        'best_recall': best_recall,   # FIX: was best_accuracy
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

    # Feature importance — extract from the inner classifier inside ImbPipeline.
    # preprocessor.get_feature_names_out() handles the nested sub-pipelines
    # (numeric_pipeline / categorical_pipeline) added in build_preprocessor().
    # Strip the 'num__' / 'cat__' prefix for readability.
    try:
        inner_clf = best_model.named_steps['classifier']

        raw_names = preprocessor.get_feature_names_out()
        feature_names = [n.split('__', 1)[1] for n in raw_names]

        if hasattr(inner_clf, 'feature_importances_'):
            importances = inner_clf.feature_importances_
        elif hasattr(inner_clf, 'coef_'):
            importances = np.abs(inner_clf.coef_[0])
        else:
            raise AttributeError("No feature_importances_ or coef_ on classifier")

        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        print("\nTop 15 Feature Importances:")
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
