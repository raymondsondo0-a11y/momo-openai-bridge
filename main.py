
import os
import requests
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from google import genai

app = FastAPI()

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MOMO_API_TOKEN = os.getenv("MOMO_API_TOKEN")

# ============================================================
# AI CLIENTS
# ============================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# SETTINGS
# ============================================================

OPENAI_MODEL = "gpt-5.6-luna"
GEMINI_MODEL = "gemini-3.7-flash"

SYSTEM_INSTRUCTIONS = (
    "You are a helpful WhatsApp customer service assistant. "
    "Reply naturally, briefly and professionally. "
    "Use the same language as the customer. "
    "If the customer writes Swahili, reply in Swahili."
)

MOMO_SEND_URL = "https://business.momo.tz/api/v3/whatsapp/send"


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Momo AI Bridge is running",
        "primary": "OpenAI",
        "backup": "Gemini"
    }


# ============================================================
# OPENAI
# ============================================================

def ask_openai(customer_message: str):

    print("AI PROVIDER: OpenAI")

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=customer_message
    )

    reply = response.output_text.strip()

    if not reply:
        raise Exception("OpenAI returned an empty response")

    return reply


# ============================================================
# GEMINI BACKUP
# ============================================================

def ask_gemini(customer_message: str):

    print("AI PROVIDER: Gemini BACKUP")

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=customer_message,
        config={
            "system_instruction": SYSTEM_INSTRUCTIONS,
            "temperature": 0.7,
            "max_output_tokens": 500
        }
    )

    reply = response.text.strip()

    if not reply:
        raise Exception("Gemini returned an empty response")

    return reply


# ============================================================
# AI FAILOVER SYSTEM
# ============================================================

def get_ai_reply(customer_message: str):

    # --------------------------------------------------------
    # TRY OPENAI FIRST
    # --------------------------------------------------------

    try:

        return ask_openai(customer_message)

    except Exception as openai_error:

        print("OPENAI FAILED:", str(openai_error))
        print("SWITCHING TO GEMINI BACKUP...")


    # --------------------------------------------------------
    # OPENAI FAILED → TRY GEMINI
    # --------------------------------------------------------

    try:

        return ask_gemini(customer_message)

    except Exception as gemini_error:

        print("GEMINI FAILED:", str(gemini_error))

        raise Exception(
            "Both AI providers failed"
        )


# ============================================================
# MOMO WEBHOOK
# ============================================================

@app.post("/momo/webhook")
async def momo_webhook(request: Request):

    data = await request.json()

    print("MOMO WEBHOOK:", data)

    # --------------------------------------------------------
    # ONLY PROCESS INCOMING MESSAGES
    # --------------------------------------------------------

    if data.get("event") != "message.received":

        return {
            "status": "ignored"
        }

    # --------------------------------------------------------
    # GET CUSTOMER DATA
    # --------------------------------------------------------

    customer_number = data.get("sender")
    customer_message = data.get("body")

    if not customer_number or not customer_message:

        return {
            "status": "ignored"
        }

    print("CUSTOMER:", customer_number)
    print("MESSAGE:", customer_message)

    # --------------------------------------------------------
    # GET AI RESPONSE WITH AUTOMATIC FAILOVER
    # --------------------------------------------------------

    try:

        ai_reply = get_ai_reply(customer_message)

        print("AI REPLY:", ai_reply)

    except Exception as e:

        print("ALL AI PROVIDERS FAILED:", str(e))

        # Do not crash the webhook with a useless message.
        # Return a controlled error.

        raise HTTPException(
            status_code=503,
            detail="AI providers temporarily unavailable"
        )

    # --------------------------------------------------------
    # SEND RESPONSE BACK TO MOMO
    # --------------------------------------------------------

    headers = {
        "Authorization": f"Bearer {MOMO_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "recipient": customer_number,
        "message": ai_reply
    }

    try:

        momo_response = requests.post(
            MOMO_SEND_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        print(
            "MOMO SEND:",
            momo_response.status_code
        )

        print(
            momo_response.text
        )

        if not momo_response.ok:

            raise HTTPException(
                status_code=502,
                detail="Momo message send failed"
            )

    except requests.RequestException as e:

        print(
            "MOMO SEND ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=502,
            detail="Momo connection failed"
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {
        "status": "success",
        "reply_sent": True
    }
