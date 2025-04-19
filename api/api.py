from fastapi import FastAPI, HTTPException
from src.classifier import EmailClassifier
from src.masking import mask_pii  # Correct import path
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

# Load model on startup
MODEL_PATH = Path("models/best_email_classifier.pkl")
classifier = EmailClassifier(MODEL_PATH)

class EmailRequest(BaseModel):
    email_body: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Email Classifier API!"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}

@app.post("/classify")
async def classify_email(request: EmailRequest):
    if not request.email_body or not isinstance(request.email_body, str):
        raise HTTPException(status_code=400, detail="Invalid email_body. It must be a non-empty string.")
    
    # Mask the email
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def demask(masked_email, masked_entities):
    for entity in masked_entities:
        start, end = entity['position']
        masked_email = masked_email[:start] + entity['entity'] + masked_email[end:]
    return masked_email