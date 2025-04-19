import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.masking import mask_pii  # Import the mask_pii function

app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str

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

classifier = EmailClassifier()

@app.post("/classify")
async def classify_email(request: EmailRequest):
    if not request.email_body or not isinstance(request.email_body, str):
        raise HTTPException(status_code=400, detail="Invalid email_body. It must be a non-empty string.")
    
    # Mask the email using the imported mask_pii function
    masked_email, masked_entities = mask_pii(request.email_body)
    
    # Predict the category using the masked email
    predicted_label = classifier.predict([masked_email])[0]
    
    # Return the response in the required format
    return {
        "input_email_body": request.email_body,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_label
    }

if __name__ == "__main__":
    clf = EmailClassifier()

    try:
        # Load real data
        df = pd.read_csv(r"D:\email_classifier_project\data\combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv")
        test_email = df.sample(1).iloc[0]["email"]

        print("\n🔍 Sample Email:\n", test_email)
        
        # Mask the email
        masked_email, masked_entities = mask_pii(test_email)
        print("\n🔒 Masked Email:\n", masked_email)
        print("\n📋 Masked Entities:\n", masked_entities)
        
        # Predict the category
        predicted_label = clf.predict([masked_email])[0]
        print("\n📌 Predicted Category:", predicted_label)

    except Exception as e:
        print(f"❌ Error: {e}")
