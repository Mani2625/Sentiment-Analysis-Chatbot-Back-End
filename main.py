from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, traceback
from google import genai
from google.genai import types

app = Flask(__name__)

# ✅ Define your frontend URL
FRONTEND_URL = "https://sentiment-analysis-210161969755.asia-south1.run.app"

# ✅ Allow only your frontend
CORS(app, resources={r"/api/*": {"origins": [FRONTEND_URL]}}, supports_credentials=True)

try:
    client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
    print("✅ Gemini client initialized")
except Exception as e:
    print(f"⚠️ Gemini init failed: {e}")
    client = None


@app.after_request
def add_cors_headers(response):
    """Ensure every response has CORS headers"""
    response.headers.add("Access-Control-Allow-Origin", FRONTEND_URL)
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_endpoint():
    # ✅ Handle OPTIONS preflight requests
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight success"}), 200

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    text = data["message"]

    if not client:
        return jsonify({
            "sentiment": "ERROR",
            "sentiment_emoji": "❌",
            "chatbot_response": "Gemini API not initialized"
        }), 500

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Analyze sentiment of: \"{text}\" and return JSON with keys sentiment, emoji, response.",
            config=types.GenerateContentConfig(
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
        res = json.loads(response.text)
        return jsonify({
            "user_message": text,
            "sentiment": res.get("sentiment"),
            "sentiment_emoji": res.get("emoji"),
            "chatbot_response": res.get("response")
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "sentiment": "ERROR",
            "sentiment_emoji": "❌",
            "chatbot_response": f"Model error: {e}"
        }), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running OK"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
