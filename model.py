import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_crop_model_real_data():
    """Train crop recommendation model with real Kaggle data"""
    
    try:
        # Download dataset from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
        # Save as 'Crop_recommendation.csv' in the same directory
        
        df = pd.read_csv('Crop_recommendation.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\nColumn names:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for the correct label column name
        # Common names: 'label', 'crop', 'Crop'
        label_column = 'label' if 'label' in df.columns else 'crop'
        
        # Prepare features and target
        X = df.drop(label_column, axis=1)
        y = df[label_column]
        
        print(f"\nUnique crops: {y.nunique()}")
        print("Crop distribution:")
        print(y.value_counts())
        
    except FileNotFoundError:
        print("CSV file not found. Using synthetic data instead...")
        print("Download dataset from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        return train_crop_model_synthetic()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✓ Model saved as 'crop_model.pkl'")
    print("✓ Training complete!")

def train_crop_model_synthetic():
    """Fallback function with synthetic data"""
    crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Chickpea', 'Banana', 'Mango']
    n_samples = 1000
    
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(10, 45, n_samples),
        'humidity': np.random.uniform(20, 100, n_samples),
        'ph': np.random.uniform(4, 10, n_samples),
        'rainfall': np.random.uniform(20, 300, n_samples),
        'label': np.random.choice(crops, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Synthetic model saved successfully!")

if __name__ == '__main__':
    train_crop_model_real_data()
