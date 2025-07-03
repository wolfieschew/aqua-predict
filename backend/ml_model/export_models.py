import joblib
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def export_models():
    try:
        # Load dataset
        df = pd.read_csv('waterQuality1.csv')
        logging.info("Dataset loaded successfully")

        # Preprocessing
        df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
        df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')
        df = df.dropna()

        # Split features and target
        X = df.drop(['is_safe'], axis=1)
        y = df['is_safe']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Scale features
        scaler = MinMaxScaler(feature_range=(-10, 10))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply SMOTE
        oversample = SMOTE(k_neighbors=3, random_state=100)
        X_smote, y_smote = oversample.fit_resample(X_train, y_train)

        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
        logging.info("Scaler saved successfully")

        # Load and save models from classifiers
        models = {
            'random_forest': classifier,  # from Random Forest section
            'neural_network': classifier, # from Neural Network section
            'decision_tree': classifier,  # from Decision Tree section
            'knn': classifier,           # from KNN section
            'xgboost': classifier        # from XGBoost section
        }

        for name, model in models.items():
            joblib.dump(model, f'{name}_model.pkl')
            logging.info(f"{name} model saved successfully")

        return True

    except Exception as e:
        logging.error(f"Error during export: {str(e)}")
        return False

if __name__ == "__main__":
    if export_models():
        print("\nAll models exported successfully!")
    else:
        print("\nError exporting models. Check the logs above.")