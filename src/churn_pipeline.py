# FILE: src/churn_pipeline.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

class ChurnModelPipeline:
    """
    A class to encapsulate the entire churn prediction workflow, from data
    preprocessing to model training and prediction.
    """
    def __init__(self, model_type='xgb'):
        # Initialize attributes that will be learned during training
        self.model_type = model_type
        self.model = None
        self.best_params_ = None
        self.feature_columns_ = None

    def _preprocess(self, crm_df, tickets_df, is_training=True):
        """Private method for all data preprocessing steps."""
        # Date handling
        tickets_df['Date_Created'] = pd.to_datetime(tickets_df['Date_Created'])
        
        # Ticket aggregation
        tickets_agg_df = tickets_df.groupby('CustomerID').agg(
            Total_tickets=('TicketID', 'count'),
            Avg_res_time=('Resolution_Time_Hours', 'mean'),
            Reopened_tkt_count=('Was_Reopened', 'sum')
        ).reset_index()

        # Recent ticket calculation
        cur_dt = datetime(2025, 8, 25)
        ninety_days = cur_dt - timedelta(days=90)
        rec_tkt = tickets_df[tickets_df['Date_Created'] >= ninety_days]
        rec_tkt_cnt = rec_tkt.groupby('CustomerID').size().reset_index(name='Tickets_last_90_days')
        tickets_agg_df = pd.merge(tickets_agg_df, rec_tkt_cnt, on='CustomerID', how='left')

        # One-hot encode ticket categories
        category_dummies = pd.get_dummies(tickets_df[['CustomerID', 'Ticket_Category']], columns=['Ticket_Category'], prefix='Count')
        category_counts = category_dummies.groupby('CustomerID').sum().reset_index()
        tickets_agg_df = pd.merge(tickets_agg_df, category_counts, on='CustomerID', how='left')
        
        # Merge CRM and ticket data
        final_df = pd.merge(crm_df, tickets_agg_df, on='CustomerID', how='left')
        
        # Fill NaNs created from the merge
        fill_zero_cols = tickets_agg_df.columns.drop('CustomerID')
        final_df[fill_zero_cols] = final_df[fill_zero_cols].fillna(0)
        
        # Convert appropriate columns to integer
        cols_to_convert = [col for col in fill_zero_cols if 'Count' in col or 'Tickets' in col]
        final_df[cols_to_convert] = final_df[cols_to_convert].astype(int)

        # Final feature engineering
        final_df['Has_Bundled_Services'] = final_df['Has_Bundled_Services'].astype(int)
        if 'Churned' in final_df.columns:
            final_df['Churned'] = final_df['Churned'].astype(int)
        
        final_df = pd.get_dummies(final_df, columns=['Contract_Type', 'Recent_Plan_Change'], drop_first=True)
        
        # Store feature column names during training
        if is_training:
            self.feature_columns_ = [col for col in final_df.columns if col not in ['CustomerID', 'Churned', 'Customer_Location_Postcode']]
        
        # Ensure new data has the same columns as training data for consistency
        processed_data = final_df.reindex(columns=self.feature_columns_, fill_value=0)
        
        # Re-add target if it exists for the training process
        if 'Churned' in final_df.columns:
            processed_data = pd.concat([processed_data, final_df['Churned']], axis=1)

        return processed_data

    def train(self, crm_df, tickets_df):
        """Trains the model on the provided dataframes."""
        print("--- Starting Training ---")
        processed_df = self._preprocess(crm_df, tickets_df, is_training=True)

        X = processed_df[self.feature_columns_]
        y = processed_df['Churned']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Apply SMOTE to handle class imbalance
        minority_class_count = y_train.value_counts().min()
        k_neighbors = min(5, minority_class_count - 1) if minority_class_count > 1 else 1
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Select model and define hyperparameter grid
        if self.model_type == 'xgb':
            model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        else: # 'rf'
            model = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
            
        # Perform grid search to find the best model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        print(f"\nBest Parameters Found: {self.best_params_}")
        
        # Evaluate the final model on the hold-out test set
        y_pred = self.model.predict(X_test)
        print("\n--- Evaluation on Test Set ---")
        print(classification_report(y_test, y_pred))

    def predict(self, crm_df_new, tickets_df_new):
        """Makes predictions on new, unseen data."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Please call .train() first.")
        
        processed_df = self._preprocess(crm_df_new, tickets_df_new, is_training=False)
        X_new = processed_df[self.feature_columns_]
        
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)[:, 1]
        
        return predictions, probabilities

    def save(self, filepath):
        """Saves the entire pipeline object to a file."""
        print(f"Saving pipeline to {filepath}...")
        joblib.dump(self, filepath)
        print("✅ Pipeline saved.")

    @staticmethod
    def load(filepath):
        """Loads a pipeline object from a file."""
        print(f"Loading pipeline from {filepath}...")
        pipeline = joblib.load(filepath)
        print("✅ Pipeline loaded.")
        return pipeline