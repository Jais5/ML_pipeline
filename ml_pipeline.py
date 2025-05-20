from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def generate_data(samples=500, features=10, informative=5, redundant=2, classes=2):
    """
    Generate synthetic classification data.
    """
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=informative,
        n_redundant=redundant,
        n_classes=classes,
        random_state=42
    )
    return X, y

def split_data(X, y, test_size=0.25):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_model(X_train, y_train):
    """
    Train a RandomForest classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def feature_importance(clf, feature_names=None):
    """
    Display feature importance from the model.
    """
    importances = clf.feature_importances_
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

def main():
    print("Generating synthetic data...")
    X, y = generate_data()

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training RandomForest classifier...")
    clf = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test, y_test)

    print("Showing feature importances...")
    feature_importance(clf)

if __name__ == "__main__":
    main()
