import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)

# ✅ Allow only your frontend domain
FRONTEND_URL = "https://sentiment-analysis-210161969755.asia-south1.run.app"

# ✅ CORS configuration
CORS(app, resources={r"/api/*": {"origins": [FRONTEND_URL]}}, supports_credentials=True)

# --- Gemini Initialization ---
try:
    client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
    print(f"✅ Gemini client initialized with model: {GEMINI_MODEL}")
except Exception as e:
    print(f"⚠️ WARNING: Gemini client initialization failed: {e}")
    client = None


def get_gemini_response(text):
    """Get sentiment and reply from Gemini."""
    if client is None:
        return "ERROR", "❌", "Gemini client not initialized. Check API key."

    system_instruction = (
        "You are a friendly chatbot that analyzes sentiment and responds nicely. "
        "Always respond in JSON with keys: sentiment, emoji, response."
    )
    prompt = f"Analyze this text and respond in JSON: \"{text}\""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "emoji": {"type": "string"},
                        "response": {"type": "string"}
                    },
                    "required": ["sentiment", "emoji", "response"]
                }
            )
        )
        json_data = json.loads(response.text)
        return json_data["sentiment"], json_data["emoji"], json_data["response"]

    except Exception as e:
        traceback.print_exc()
        return "ERROR", "❌", f"Model error: {e}"


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_endpoint():
    """Main chat endpoint with CORS preflight handling."""
    # ✅ Handle OPTIONS preflight requests
    if request.method == "OPTIONS":
        response = jsonify({"status": "CORS preflight success"})
        response.headers.add("Access-Control-Allow-Origin", FRONTEND_URL)
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    # ✅ Handle POST requests
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"]
    sentiment, emoji, reply = get_gemini_response(user_message)

    response = jsonify({
        "user_message": user_message,
        "sentiment": sentiment,
        "sentiment_emoji": emoji,
        "chatbot_response": reply
    })
    response.headers.add("Access-Control-Allow-Origin", FRONTEND_URL)
    return response, 200


@app.route("/", methods=["GET"])
def home():
    """Health check route."""
    return jsonify({"status": "Backend running", "region": "asia-south1"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
