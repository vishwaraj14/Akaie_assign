import pandas as pd
from src.classifier import EmailClassifier
from src.masking import mask_pii  # Import the mask_pii function

if __name__ == "__main__":
    clf = EmailClassifier("models/best_email_classifier.pkl")

    try:
        # Load real data
        df = pd.read_csv(r"D:\email_classifier_project\data\combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv")
        
        # Ensure the 'email' column exists
        if "email" not in df.columns:
            raise Exception("The 'email' column is missing in the CSV file.")
        
        # Sample a random email
        test_email = df.sample(1).iloc[0]["email"]

        print("\n🔍 Sample Email:\n", test_email)
        
        # Mask the email
        masked_email, masked_entities = mask_pii(test_email)
        print("\n🔒 Masked Email:\n", masked_email)
        print("\n📋 Masked Entities:\n", masked_entities)
        
        # Predict the category
        predicted_label = clf.predict([masked_email])[0]
        print("\n📌 Predicted Category:", predicted_label)

    except FileNotFoundError:
        print("❌ Error: The CSV file was not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error: {e}")