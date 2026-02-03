import json
from pathlib import Path
from datetime import datetime
import smtplib
from email.message import EmailMessage
from difflib import SequenceMatcher

DEFAULT_RESPONSE = [
    "Hey hey, off-topic alert 🚨 I only talk Domesan stuff here",
    "That question is cool, but my brain is strictly portfolio-trained 🤖",
    "I’m on portfolio duty right now — ask me something about him 😄",
    "My knowledge license only covers my manager data, nothing else 😅",
    "Let’s keep it portfolio-centric, shall we?"
]

JAILBREAK_ATTEMPT_RESPONSE = [
    "Hey bruhh, you tryna jailbreak me? I am ahead of you already",
    "Nice try 😏 but I saw that move coming from a mile away",
    "Bold attempt, but I run on rules, not loopholes",
    "You’re trying to outsmart me… I helped write the test 😌",
    "That trick worked in 2022, not today my friend"
]


DATA_SHORTAGE_RESPONSE = [
    "Sorry man, I cannot answer your query"
]

HISTORY_FILE = Path("question_history.json")


def split_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def convert_ai_response_to_valid_json(ai_response):
    if "```" in ai_response:
        ai_response = ai_response.split("```")[1]
        if ai_response.strip().startswith("json"):
            ai_response = ai_response.strip()[4:]
    return ai_response

def is_similar(new_q, existing_q, threshold=0.85):
    """
    Returns True if similarity ratio is above threshold (0.85 means 85% similar).
    """
    # Convert to lowercase to make it case-insensitive
    return SequenceMatcher(None, new_q.lower(), existing_q.lower()).ratio() > threshold

def store_question(question: str):
    # 1. Load existing data
    if not HISTORY_FILE.exists():
        data = {"questions": []}
    else:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {"questions": []}

    # 2. Smart Check: Iterate through existing questions
    # We check the new question against EVERY existing question
    for entry in data["questions"]:
        existing_q = entry["question"]
        
        # Exact match (fastest check)
        if existing_q == question:
            print(f"Skipping: Exact duplicate found.")
            return

        # Similarity check (smart check)
        if is_similar(question, existing_q):
            print(f"Skipping: Similar question found: '{existing_q}'")
            return

    # 3. Append if no duplicates found
    data["questions"].append({
        "question": question,
        "timestamp": datetime.utcnow().isoformat()
    })

    # 4. Write back
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print("Question saved.")

def get_all_questions():
    if not HISTORY_FILE.exists():
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("questions", [])


def send_failure_mail(error_message: str):
    import os
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL')
    PASSWORD = os.getenv('PASSWORD')
    
    msg = EmailMessage()
    msg["Subject"] = "SmartFolio 🚨 Model Failure Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.set_content(f"""
Model failure detected.

Error details:
{error_message}
""")

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, PASSWORD)
        server.send_message(msg)
