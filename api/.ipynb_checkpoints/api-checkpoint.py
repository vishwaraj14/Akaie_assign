from fastapi import FastAPI
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
    # Mask the email
    masked_email, masked_entities = mask_pii(request.email_body)
    
    # Predict the category using the masked email
    predicted_label = classifier.predict([masked_email])[0]
    
    return {
        "input_email_body": request.email_body,
        "masked_email": masked_email,
        "list_of_masked_entities": masked_entities,
        "predicted_label": predicted_label
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}