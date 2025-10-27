# main.py - Modified for Gemini API

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
# CRITICAL: In a real deployment, replace the wildcard with your frontend's URL!
# We keep the wildcard for demonstration, but allow credentials for better browser compatibility.
CORS(app, resources={r"/api/*": {"origins": "*", "allow_credentials": True}}) 

# --- Gemini Initialization ---

# 1. Initialize the client (API key must be set as GEMINI_API_KEY environment variable in Cloud Run)
try:
    client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash" 
    print(f"Gemini client initialized with model: {GEMINI_MODEL}")
except Exception as e:
    # This serves as a fail-safe if the key isn't set
    print(f"WARNING: Gemini client initialization failed: {e}")
    client = None

# --- Sentiment Analysis and Response Generation using Gemini ---

def get_gemini_response(text):
    """
    Sends the user text to Gemini for sentiment analysis and a conversational response.
    Returns (sentiment_label, response_emoji, chatbot_response).
    """
    if client is None:
        return "ERROR", "❌", "Model Error: Gemini API client not initialized. Check GEMINI_API_KEY."

    # System instruction to guide the model to perform two tasks and output JSON
    system_instruction = (
        "You are a helpful and friendly sentiment analysis chatbot. "
        "Your task is to analyze the user's message and provide a response in a single JSON object. "
        "The JSON MUST contain the following keys: "
        "'sentiment' (must be 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'), "
        "'emoji' (a single emoji representing the sentiment), and "
        "'response' (a short, conversational reply acknowledging the sentiment and responding to the user's message)."
    )
    
    prompt = f"Analyze the following text and provide your output as a single, valid JSON object. Text to analyze: \"{text}\""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # Force JSON output for reliable parsing
                response_mime_type="application/json", 
                response_schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                        "emoji": {"type": "string"},
                        "response": {"type": "string"}
                    },
                    "required": ["sentiment", "emoji", "response"]
                }
            )
        )
        
        # Parse the JSON response text
        json_data = json.loads(response.text)
        
        sentiment_label = json_data.get('sentiment', 'NEUTRAL').upper()
        sentiment_emoji = json_data.get('emoji', '😐')
        chatbot_response = json_data.get('response', 'I received your message but could not generate a proper response.')

        return sentiment_label, sentiment_emoji, chatbot_response
        
    except Exception as e:
        print(f"Gemini API call or JSON parsing failed: {e}")
        return "ERROR", "❌", "Failed to connect to the model or parse its response. Check logs."


# --- Flask API Endpoint ---

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """ Handles the chat message request. """
    data = request.get_json(silent=True)
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Missing "message" key.'}), 400
        
    user_message = data['message']

    # Analyze Sentiment and get conversational response from Gemini
    sentiment_label, sentiment_emoji, bot_response = get_gemini_response(user_message)

    # Return Final JSON Response
    return jsonify({
        'user_message': user_message,
        'sentiment': sentiment_label,
        'sentiment_emoji': sentiment_emoji,
        'chatbot_response': bot_response
    })

# The following is for local testing only. Gunicorn handles production.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)