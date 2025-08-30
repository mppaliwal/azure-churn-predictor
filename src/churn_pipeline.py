import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

class ChurnModelPipeline:
    """
    A class to encapsulate the entire churn prediction workflow, from data
    preprocessing to model training, prediction, and explanation.
    """
    def __init__(self, model_type='xgb'):
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

        # Ticket category one-hot encoding
        category_dummies = pd.get_dummies(tickets_df[['CustomerID', 'Ticket_Category']], columns=['Ticket_Category'], prefix='Count')
        category_counts = category_dummies.groupby('CustomerID').sum().reset_index()
        tickets_agg_df = pd.merge(tickets_agg_df, category_counts, on='CustomerID', how='left')
        
        # Merge CRM and ticket data
        final_df = pd.merge(crm_df, tickets_agg_df, on='CustomerID', how='left')
        
        # Fill NaNs created from the merge
        fill_zero_cols = tickets_agg_df.columns.drop('CustomerID')
        final_df[fill_zero_cols] = final_df[fill_zero_cols].fillna(0)
        
        # Convert data types
        cols_to_convert_to_int = [col for col in fill_zero_cols if 'Count' in col or 'Tickets' in col]
        final_df[cols_to_convert_to_int] = final_df[cols_to_convert_to_int].astype(int)
        final_df['Has_Bundled_Services'] = final_df['Has_Bundled_Services'].astype(int)
        if 'Churned' in final_df.columns:
            final_df['Churned'] = final_df['Churned'].astype(int)
        
        final_df = pd.get_dummies(final_df, columns=['Contract_Type', 'Recent_Plan_Change'], drop_first=True)
        
        if is_training:
            self.feature_columns_ = [col for col in final_df.columns if col not in ['CustomerID', 'Churned', 'Customer_Location_Postcode']]
        
        for col in self.feature_columns_:
            if col not in final_df.columns:
                final_df[col] = 0
        
        return final_df

    def train(self, crm_df, tickets_df):
        """Trains the model on the provided data."""
        print("--- Starting Training ---")
        processed_df = self._preprocess(crm_df, tickets_df, is_training=True)

        X = processed_df[self.feature_columns_]
        y = processed_df['Churned']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        minority_class_count = y_train.value_counts().min()
        k_neighbors = min(5, minority_class_count - 1)
        if k_neighbors < 1:
            X_train_resampled, y_train_resampled = X_train, y_train
        else:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        if self.model_type == 'xgb':
            model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        else: # rf
            model = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [1, 2]}
            
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        print(f"\nBest Parameters Found: {self.best_params_}")
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
        """Saves the entire pipeline object."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """Loads a pipeline object from a file."""
        return joblib.load(filepath)
