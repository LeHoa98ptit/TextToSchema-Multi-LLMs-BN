import os
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

def main():
    # 1. Define paths to the dataset
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "dataset/relationship_training_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please run prepare_relation_data.py first.")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Prepare features (X) and target (y)
    feature_cols = ['x_text', 'x_dep', 'x_type', 'x_cooccur']
    X = df[feature_cols]
    y = df['label']
    
    # Handle NaN values (if any)
    X = X.fillna(0)
    
    # 3. Split data into train and test sets
    # Using stratify=y ensures the train and test sets have the same proportion of class labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 4. Apply SMOTE & Hyperparameter Tuning
    if HAS_SMOTE:
        print("\nApplying SMOTE to balance the training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        print("\nWarning: 'imbalanced-learn' not installed. Skipping SMOTE.")
        print("Run 'pip install imbalanced-learn' for better balancing.")
        X_train_resampled, y_train_resampled = X_train, y_train
        
    print("Tuning hyperparameters to maximize accuracy...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=2000), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    model = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    # 5. Evaluate the model
    print("\n--- Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # 6. Extract the learned parameters
    w_text, w_dep, w_type, w_cooccur = model.coef_[0]
    bias = model.intercept_[0]

    print("\n--- Learned Parameters ---")
    print(f"Weight for x_text   (w_text):   {w_text:.4f}")
    print(f"Weight for x_dep    (w_dep):    {w_dep:.4f}")
    print(f"Weight for x_type   (w_type):   {w_type:.4f}")
    print(f"Weight for x_cooccur (w_cooccur): {w_cooccur:.4f}")
    print(f"Bias                (bias):     {bias:.4f}")

    # Display the probability formula
    print("\nProbability formula P(E1, E2):")
    print(f"P = sigmoid({w_text:.4f}*x_text + {w_dep:.4f}*x_dep + {w_type:.4f}*x_type + {w_cooccur:.4f}*x_cooccur + {bias:.4f})")

    # 7. Save parameters to a JSON file for use in the project's logic
    params = {
        "w_text":    float(w_text),
        "w_dep":     float(w_dep),
        "w_type":    float(w_type),
        "w_cooccur": float(w_cooccur),
        "bias":      float(bias)
    }
    
    out_path = os.path.join(current_dir, "dataset/relationship_parameters.json")
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)
        
    print(f"\nSaved parameters to: {out_path}")

if __name__ == '__main__':
    main()