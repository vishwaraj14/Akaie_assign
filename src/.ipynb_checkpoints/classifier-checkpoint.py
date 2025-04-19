import pandas as pd
import joblib

class EmailClassifier:
    def __init__(self, model_path='models/best_email_classifier.pkl'):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise Exception(f"Model file not found at {model_path}. Please check the path.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}")

    def predict(self, email_texts):
        # Ensure input is a list of strings
        if not isinstance(email_texts, list) or not all(isinstance(email, str) for email in email_texts):
            raise ValueError("Input email_texts must be a list of strings.")
        try:
            predictions = self.model.predict(email_texts)
            return predictions
        except Exception as e:
            raise Exception(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    clf = EmailClassifier()

    try:
        # Load real data
        df = pd.read_csv(r"D:\email_classifier_project\data\combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv")
        test_email = df.sample(1).iloc[0]["email"]

        print("\n🔍 Sample Email:\n", test_email)
        print("📌 Predicted Category:", clf.predict([test_email])[0])

    except Exception as e:
        print(f"❌ Error: {e}")
