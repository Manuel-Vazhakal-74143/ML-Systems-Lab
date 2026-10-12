import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    
    # Split 70/30 stratified with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Train Logistic Regression baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    # Train Random Forest baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # Record test accuracy to 4 decimals
    results = pd.DataFrame({
        "Model": ["LogisticRegression", "RandomForestClassifier"],
        "Accuracy": [f"{lr_acc:.4f}", f"{rf_acc:.4f}"]
    })
    
    # Save to results folder
    output_path = "lab01/results/baseline_accuracy.csv"
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()