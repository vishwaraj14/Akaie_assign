import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/classify"

# Streamlit app
st.title("Email Classifier")
st.write("This app masks PII in emails and classifies them into categories.")

# Input email text
email_body = st.text_area("Enter the email text:", height=200)

if st.button("Classify Email"):
    if email_body.strip():
        # Send the email to the FastAPI endpoint
        payload = {"email_body": email_body}
        try:
            response = requests.post(FASTAPI_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Results")
                st.write("### Input Email:")
                st.write(result["input_email_body"])
                st.write("### Masked Email:")
                st.write(result["masked_email"])
                st.write("### Detected PII Entities:")
                for entity in result["list_of_masked_entities"]:
                    st.write(f"- **{entity['classification']}**: {entity['entity']} (Position: {entity['position']})")
                st.write("### Predicted Category:")
                st.write(f"**{result['predicted_label']}**")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some email text.")

st.write("---")
st.write("Developed with ❤️ using FastAPI and Streamlit.")