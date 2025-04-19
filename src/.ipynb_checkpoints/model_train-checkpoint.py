# model_train.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
from masking import mask_pii
# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {len(df)} emails")
    assert {'email', 'type'}.issubset(df.columns), "Missing required columns"

    # Mask sensitive information
    df['masked_email'], df['masked_entities'] = zip(*df['email'].apply(mask_pii))

    print("\nClass Distribution:")
    print(df['type'].value_counts(normalize=True))
    return df['masked_email'], df['type']

def evaluate_classifiers(X, y):
    classifiers = {
        'Naive Bayes': (MultinomialNB(), {
            'clf__alpha': [0.5, 1.0]
        }),
        'SVM': (LinearSVC(class_weight='balanced'), {
            'clf__C': [0.1, 1.0]
        }),
        'Random Forest': (RandomForestClassifier(class_weight='balanced'), {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 20]
        }),
        'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000), {
            'clf__C': [0.1, 1.0]
        })
    }

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)

    results = []
    for name, (model, params) in classifiers.items():
        print(f"\nEvaluating {name}...")
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', model)
        ])

        grid = GridSearchCV(pipeline, params, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X, y)

        # Final evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        best_model = grid.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        results.append({
            'classifier': name,
            'cv_mean_f1': grid.best_score_,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred, average='weighted'),
            'model': best_model
        })

        print(f"Best CV F1: {grid.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

    return pd.DataFrame(results)

def save_best_model(results_df):

    # Ensure the models folder exists
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)

    # Save the best model
    best_model = results_df.loc[results_df['test_f1'].idxmax()]
    print(f"\nBest model: {best_model['classifier']}")
    print(f"Test F1: {best_model['test_f1']:.4f}")
    model_path = os.path.join(models_folder, 'best_email_classifier.pkl')
    joblib.dump(best_model['model'], model_path)
    print(f"Best model saved as '{model_path}'")
    return best_model['model']

def main():
    try:
        X, y = load_data(r"D:\email_classifier_project\data\combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv")
        results = evaluate_classifiers(X, y)
        save_best_model(results)
        print("\n✅ Model training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
