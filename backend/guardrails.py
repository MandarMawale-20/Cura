import re

EMERGENCY_PATTERNS = [
    r"chest pain",
    r"can'?t breathe",
    r"trouble breathing",
    r"difficulty breathing",
    r"severe bleeding",
    r"heavy bleeding",
    r"unconscious",
    r"unresponsive",
    r"stroke",
    r"seizure",
    r"suicidal",
    r"kill myself",
    r"overdose",
    r"heart attack",
    r"choking",
    r"severe allergic reaction",
    r"anaphylaxis",
]

DIAGNOSIS_PATTERNS = [
    r"do i have (\w+\s?){1,4}",
    r"am i having a? ?(\w+\s?){1,4}",
    r"what disease do i have",
    r"is this cancer",
    r"diagnose me",
    r"what'?s wrong with me",
]

EMERGENCY_MESSAGE = (
    "This sounds like it could be a medical emergency. Please call your local "
    "emergency number or go to the nearest emergency room right away. I'm not "
    "able to provide guidance for urgent or life-threatening situations."
)

DIAGNOSIS_MESSAGE = (
    "I can share general health information, but I can't diagnose medical "
    "conditions. For an accurate diagnosis, please see a licensed healthcare "
    "professional who can properly examine and evaluate your symptoms."
)

DISCLAIMER = (
    "This information is for general education only and isn't a substitute "
    "for professional medical advice. Please consult a doctor for concerns "
    "specific to your health."
)


def _matches_any(text: str, patterns: list) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def check_emergency(message: str) -> bool:
    return _matches_any(message, EMERGENCY_PATTERNS)


def check_diagnosis_request(message: str) -> bool:
    return _matches_any(message, DIAGNOSIS_PATTERNS)
