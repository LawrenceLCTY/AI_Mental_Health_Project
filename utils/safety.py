# utils/safety.py
import re
from typing import List

SUICIDE_TERMS = ["自杀", "想死", "结束生命", "轻生", "想自杀"]
MEDICAL_DIAGNOSIS_TERMS = ["确诊", "诊断为", "属于抑郁症", "患有"]
PII_PATTERNS = [
    r"\b\d{3,}-\d{3,}-\d{4}\b",  # phone-like
    r"\b\d{11}\b",               # China phone
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
]

def heuristic_check_text(text: str) -> List[str]:
    issues = []
    t = text.lower()
    for term in SUICIDE_TERMS:
        if term in t:
            issues.append("Self-harm / suicide related term detected")
            break
    for term in MEDICAL_DIAGNOSIS_TERMS:
        if term in t:
            issues.append("Potential diagnostic/medical-prescriptive language")
            break
    for pat in PII_PATTERNS:
        if re.search(pat, text):
            issues.append("Potential PII detected (phone/email/ID)")
            break
    return issues
