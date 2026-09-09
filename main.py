
import os
import time
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MOMO_API_TOKEN = os.getenv("MOMO_API_TOKEN")

# ============================================================
# AI CLIENTS
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ============================================================
# MODELS
# ============================================================

OPENAI_MODEL = "gpt-5.6-luna"
GEMINI_MODEL = "gemini-3.7-flash"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ============================================================
# SETTINGS
# ============================================================

SYSTEM_INSTRUCTIONS = (
    "You are a helpful WhatsApp customer service assistant. "
    "Reply naturally, briefly and professionally. "
    "Use the same language as the customer. "
    "If the customer writes Swahili, reply in Swahili. "
    "Never mention OpenAI, Gemini, DeepSeek, Google, model names, "
    "API keys, providers, quotas, or internal system failures. "
    "You are simply the customer's WhatsApp assistant."
)

MOMO_SEND_URL = (
    "https://business.momo.tz/api/v3/whatsapp/send"
)

# ============================================================
# PROVIDER COOLDOWNS
# ============================================================

provider_cooldown = {
    "OpenAI": 0,
    "Gemini": 0,
    "DeepSeek": 0
}

# Default cooldowns
QUOTA_COOLDOWN = 1800       # 30 minutes
TEMPORARY_COOLDOWN = 60     # 1 minute
GENERAL_COOLDOWN = 120      # 2 minutes


def is_provider_available(provider):
    return time.time() >= provider_cooldown[provider]


def cooldown_provider(provider, seconds, reason):
    provider_cooldown[provider] = time.time() + seconds

    print(
        f"{provider} COOLDOWN: "
        f"{seconds} seconds | {reason}"
    )


def provider_status():
    now = time.time()

    status = {}

    for provider, until in provider_cooldown.items():

        if now >= until:
            status[provider] = "available"

        else:
            remaining = int(until - now)

            status[provider] = (
                f"cooldown ({remaining}s remaining)"
            )

    return status


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "Momo AI Bridge is running",
        "primary": "OpenAI",
        "backup_1": "Gemini",
        "backup_2": "DeepSeek",
        "providers": provider_status()
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
        raise Exception(
            "OpenAI returned an empty response"
        )

    return reply


# ============================================================
# GEMINI
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
        raise Exception(
            "Gemini returned an empty response"
        )

    return reply


# ============================================================
# DEEPSEEK
# ============================================================

def ask_deepseek(customer_message: str):

    print("AI PROVIDER: DeepSeek BACKUP #2")

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTIONS
            },
            {
                "role": "user",
                "content": customer_message
            }
        ],
        temperature=0.7,
        max_tokens=500,
        stream=False
    )

    reply = response.choices[0].message.content

    if not reply:
        raise Exception(
            "DeepSeek returned an empty response"
        )

    reply = reply.strip()

    if not reply:
        raise Exception(
            "DeepSeek returned an empty response"
        )

    return reply


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_error(error):

    message = str(error).lower()

    quota_words = [
        "429",
        "quota",
        "rate limit",
        "resource_exhausted",
        "requests per day",
        "limit",
        "too many requests"
    ]

    temporary_words = [
        "503",
        "502",
        "504",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "high demand",
        "overloaded"
    ]

    if any(word in message for word in quota_words):
        return "quota"

    if any(word in message for word in temporary_words):
        return "temporary"

    return "general"


# ============================================================
# SMART PROVIDER FAILURE HANDLER
# ============================================================

def handle_provider_failure(provider, error):

    error_type = classify_error(error)

    print(
        f"{provider} FAILED | "
        f"type={error_type} | "
        f"error={str(error)}"
    )

    if error_type == "quota":

        cooldown_provider(
            provider,
            QUOTA_COOLDOWN,
            "quota/rate limit"
        )

    elif error_type == "temporary":

        cooldown_provider(
            provider,
            TEMPORARY_COOLDOWN,
            "temporary service problem"
        )

    else:

        cooldown_provider(
            provider,
            GENERAL_COOLDOWN,
            "general provider error"
        )


# ============================================================
# AI FAILOVER SYSTEM
# ============================================================

def get_ai_reply(customer_message: str):

    # --------------------------------------------------------
    # PROVIDER 1 — OPENAI
    # --------------------------------------------------------

    if is_provider_available("OpenAI"):

        try:

            return ask_openai(customer_message)

        except Exception as error:

            handle_provider_failure(
                "OpenAI",
                error
            )

            print(
                "SWITCHING TO GEMINI..."
            )

    else:

        print(
            "OpenAI SKIPPED: provider is on cooldown"
        )


    # --------------------------------------------------------
    # PROVIDER 2 — GEMINI
    # --------------------------------------------------------

    if is_provider_available("Gemini"):

        try:

            return ask_gemini(customer_message)

        except Exception as error:

            handle_provider_failure(
                "Gemini",
                error
            )

            print(
                "SWITCHING TO DEEPSEEK..."
            )

    else:

        print(
            "Gemini SKIPPED: provider is on cooldown"
        )


    # --------------------------------------------------------
    # PROVIDER 3 — DEEPSEEK
    # --------------------------------------------------------

    if is_provider_available("DeepSeek"):

        try:

            return ask_deepseek(customer_message)

        except Exception as error:

            handle_provider_failure(
                "DeepSeek",
                error
            )

            print(
                "ALL AI PROVIDERS FAILED"
            )

    else:

        print(
            "DeepSeek SKIPPED: provider is on cooldown"
        )


    # --------------------------------------------------------
    # EVERYTHING FAILED
    # --------------------------------------------------------

    raise Exception(
        "All AI providers are currently unavailable"
    )


# ============================================================
# MOMO WEBHOOK
# ============================================================

@app.post("/momo/webhook")
async def momo_webhook(request: Request):

    try:

        data = await request.json()

    except Exception:

        print("INVALID MOMO JSON")

        return {
            "status": "ignored"
        }


    print(
        "MOMO WEBHOOK:",
        data
    )


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


    print(
        "CUSTOMER:",
        customer_number
    )

    print(
        "MESSAGE:",
        customer_message
    )


    # --------------------------------------------------------
    # GET AI RESPONSE
    # --------------------------------------------------------

    try:

        ai_reply = get_ai_reply(
            customer_message
        )

        print(
            "AI REPLY:",
            ai_reply
        )

    except Exception as error:

        print(
            "ALL AI PROVIDERS FAILED:",
            str(error)
        )

        # IMPORTANT:
        # Return 200 instead of 503.
        # This prevents unnecessary webhook retry storms.

        return {
            "status": "received",
            "reply_sent": False,
            "reason": "AI temporarily unavailable"
        }


    # --------------------------------------------------------
    # SEND RESPONSE TO MOMO
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

            print(
                "MOMO SEND FAILED:",
                momo_response.text
            )

            return {
                "status": "received",
                "reply_sent": False,
                "reason": "Momo send failed"
            }


    except requests.RequestException as error:

        print(
            "MOMO SEND ERROR:",
            str(error)
        )

        return {
            "status": "received",
            "reply_sent": False,
            "reason": "Momo connection failed"
        }


    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {
        "status": "success",
        "reply_sent": True
    }

