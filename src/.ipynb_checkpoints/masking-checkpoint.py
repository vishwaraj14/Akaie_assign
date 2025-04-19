import re

def mask_pii(input_text):
    patterns = {
        "full_name": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_number": r"\b\d{10}\b",
        "date_of_birth": r"\b\d{2}/\d{2}/\d{4}\b",
        "aadhar_num": r"\b\d{12}\b",
        "credit_debit_no": r"\b\d{16}\b",
        "cvv_no": r"\b\d{3}\b",
        "expiry_no": r"\b\d{2}/\d{2}\b"
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
