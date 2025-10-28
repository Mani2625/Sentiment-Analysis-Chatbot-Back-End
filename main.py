from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai

app = Flask(__name__)

# ✅ CORS setup — MUST include 'supports_credentials' and handle OPTIONS
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://sentiment-analysis-210161969755.asia-south1.run.app"],
        "supports_credentials": True
    }
})

# ✅ Gemini API setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def home():
    return jsonify({"status": "Backend running", "region": "asia-south1"})

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        # ✅ Handle CORS preflight request manually
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "https://sentiment-analysis-210161969755.asia-south1.run.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    data = request.get_json()
    user_message = data.get("message", "")

    try:
        sentiment = "POSITIVE" if "love" in user_message.lower() else "NEUTRAL"
        emoji = "😊" if sentiment == "POSITIVE" else "😐"

        prompt = f"Analyze this message's tone and respond nicely: {user_message}"
        gemini_response = model.generate_content(prompt)
        chatbot_response = gemini_response.text.strip()

        response = jsonify({
            "user_message": user_message,
            "chatbot_response": chatbot_response,
            "sentiment": sentiment,
            "sentiment_emoji": emoji
        })
        # ✅ Add CORS headers explicitly in actual response too
        response.headers.add("Access-Control-Allow-Origin", "https://sentiment-analysis-210161969755.asia-south1.run.app")
        return response, 200

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://sentiment-analysis-210161969755.asia-south1.run.app")
        return response, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
