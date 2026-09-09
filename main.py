import os
import requests
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MOMO_API_TOKEN = os.getenv("MOMO_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

MOMO_SEND_URL = "https://business.momo.tz/api/v3/whatsapp/send"


@app.get("/")
def home():
    return {"status": "Momo OpenAI Bridge is running"}


@app.post("/momo/webhook")
async def momo_webhook(request: Request):
    data = await request.json()

    print("MOMO WEBHOOK:", data)

    if data.get("event") != "message.received":
        return {"status": "ignored"}

    customer_number = data.get("sender")
    customer_message = data.get("body")

    if not customer_number or not customer_message:
        return {"status": "ignored"}

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=(
                "You are a helpful WhatsApp customer service assistant. "
                "Reply naturally, briefly and professionally. "
                "Use the same language as the customer. "
                "If the customer writes Swahili, reply in Swahili."
            ),
            input=customer_message
        )

        ai_reply = response.output_text.strip()

        headers = {
            "Authorization": f"Bearer {MOMO_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "recipient": customer_number,
            "message": ai_reply
        }

        momo_response = requests.post(
            MOMO_SEND_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        print("MOMO SEND:", momo_response.status_code)
        print(momo_response.text)

        if not momo_response.ok:
            raise HTTPException(
                status_code=502,
                detail="Momo message send failed"
            )

        return {
            "status": "success",
            "reply_sent": True
        }

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Bridge processing failed"
        )
