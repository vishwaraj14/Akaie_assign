import re

def mask_pii(input_text):
    patterns = {
        # Matches two or more capitalized words, excludes common phrases
        "full_name": r"\b(?!Dear|Customer|Support|Thank|You|As|My|Could|Additionally)[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",
        
        # Matches valid email addresses
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        
        # Matches 10-digit phone numbers, optionally with country codes and separators
        "phone_number": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        
        # Matches dates in DD/MM/YYYY, MM-DD-YYYY, or YYYY-MM-DD formats
        "date_of_birth": r"\b(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})\b",
        
        # Matches 12-digit Aadhaar numbers
        "aadhar_num": r"\b\d{12}\b",
        
        # Matches 16-digit credit/debit card numbers
        "credit_debit_no": r"\b\d{16}\b",
        
        # Matches 3-digit CVV numbers
        "cvv_no": r"\b\d{3}\b",
        
        # Matches expiry dates in MM/YY format
        "expiry_no": r"\b(0[1-9]|1[0-2])/\d{2}\b"
    }

    masked_text = input_text
    list_of_masked_entities = []

    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, masked_text):  # update to avoid shifting
            start, end = match.start(), match.end()
            entity_value = match.group()
            list_of_masked_entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": entity_value
            })
            masked_text = masked_text.replace(entity_value, f"[{entity_type}]", 1)

    return masked_text, list_of_masked_entities
