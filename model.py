import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ChurnModeling:
    def __init__(self):
        self.individual_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.trained_models = {}
        self.ensemble_pipeline = None
        self.preprocessor = None
        self.feature_names = None
        self._numeric_features = None
        self._categorical_features = None

    def _drop_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove datetime/date columns so models can train safely.
        """
        df = df.copy()

        # Remove actual datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetimetz']).columns.tolist()

        possible_date_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > 0.8 * len(df):  # 80% looks like dates
                    possible_date_cols.append(col)
            except:
                continue

        all_date_cols = list(set(datetime_cols + possible_date_cols))

        if all_date_cols:
            print(f"🗑️ Dropping datetime/date columns: {all_date_cols}")
            df.drop(columns=all_date_cols, inplace=True, errors='ignore')

        return df

    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessing pipeline after datetime columns are removed."""
        self._numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self._numeric_features),
                ('cat', categorical_transformer, self._categorical_features)
            ]
        )
        return self.preprocessor

    def train_models(self, X_train: pd.DataFrame, y_train):
        """Train ALL models: Individual + Ensemble Pipeline."""
        X_train = self._drop_datetime_columns(X_train.copy())

        print("🚀 Training Individual Models...")

        self._create_preprocessor(X_train)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        self.feature_names = self.preprocessor.get_feature_names_out()

        for name, model in self.individual_models.items():
            model.fit(X_train_processed, y_train)
            self.trained_models[name] = model
            print(f"✅ Trained {name}")

        print("🎯 Training Ensemble Pipeline...")

        ensemble_classifier = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
                ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
                ('xgb', XGBClassifier(learning_rate=0.01, random_state=42, eval_metric='logloss'))
            ],
            voting='soft'
        )

        # Re-create preprocessor for pipeline
        self._create_preprocessor(X_train)

        self.ensemble_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', ensemble_classifier)
        ])

        self.ensemble_pipeline.fit(X_train, y_train)
        self.trained_models['Ensemble Voting'] = self.ensemble_pipeline

        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"📊 Models available: {list(self.trained_models.keys())}")

    def _prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop datetime columns from any input before prediction / evaluation."""
        return self._drop_datetime_columns(X.copy())

    def evaluate_models(self, X_test: pd.DataFrame, y_test):
        """Evaluate all trained models."""
        X_test_clean = self._prepare_input(X_test)
        results = []

        for name, model in self.trained_models.items():
            try:
                if name == 'Ensemble Voting':
                    y_pred = model.predict(X_test_clean)
                else:
                    X_proc = self.preprocessor.transform(X_test_clean)
                    y_pred = model.predict(X_proc)

                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, zero_division=0)
                })
            except Exception as e:
                print(f"⚠️ Error evaluating {name}: {e}")

        eval_df = pd.DataFrame(results).round(4)
        print("📈 Model Evaluation Complete")
        return eval_df

    def get_feature_importance(self, model_name='Ensemble Voting'):
        """Get feature importance for any trained model."""
        if model_name not in self.trained_models:
            return pd.DataFrame({'Feature': ['No Model'], 'Importance': [0]})

        model = self.trained_models[model_name]

        try:
            if model_name == 'Ensemble Voting':
                rf_model = model.named_steps['classifier'].estimators_[0]
                xgb_model = model.named_steps['classifier'].estimators_[2]
                importance = (rf_model.feature_importances_ + xgb_model.feature_importances_) / 2
            elif model_name == 'Logistic Regression':
                importance = np.abs(model.coef_[0])
            else:
                importance = model.feature_importances_

            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(15)

            return feat_imp

        except Exception as e:
            print(f"⚠️ Feature importance error: {e}")
            return pd.DataFrame({'Feature': ['Error'], 'Importance': [0]})

    def predict_churn_prob(self, input_data: pd.DataFrame, model_name='Ensemble Voting'):
        """Predict churn probability for a single record."""
        if model_name not in self.trained_models:
            print(f"⚠️ Model {model_name} not trained")
            return 0.5

        input_clean = self._prepare_input(input_data)
        model = self.trained_models[model_name]

        try:
            if model_name == 'Ensemble Voting':
                prob = model.predict_proba(input_clean)[0, 1]
            else:
                input_proc = self.preprocessor.transform(input_clean)
                prob = model.predict_proba(input_proc)[0, 1]

            return float(prob)
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return 0.5
    
    def predict_churn_label(self, input_data: pd.DataFrame, model_name='Ensemble Voting', threshold=0.35):
        """
        Predict final churn class:
        0 = Stayed
        1 = Churn
        """
        prob = self.predict_churn_prob(input_data, model_name)
        return 1 if prob >= threshold else 0

        
    def get_retention_recommendations(self, probability, risk_threshold=0.5):
        """Risk-based retention recommendations."""
        if probability < 0.2:
            return "🟢 **Low Risk**: Continue standard engagement."
        elif probability < risk_threshold:
            return "🟡 **Medium Risk**: Send promotional offers & check-in emails."
        elif probability < 0.8:
            return "🟠 **High Risk**: Offer discounts, phone feedback, free upgrades."
        else:
            return "🔴 **Critical Risk**: Immediate intervention! Assign account manager."