# main.py (LLM-Powered Interactive Chatbot)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import pipeline
import google.genai as genai
from google.genai import types

# --- LLM and Sentiment Model Initialization ---

# 1. Hugging Face Model (for fast, cheap sentiment classification)
# Using the corrected, popular model identifier

hf_sentiment_pipeline = None
try:
    # Switched to the 3-class RoBERTa model for proper POSITIVE/NEUTRAL/NEGATIVE labels
    hf_sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment" # <-- Switched to 3-class model
    )
    print("✅ Hugging Face 3-Class sentiment classifier loaded.")
except Exception as e:
    print(f"❌ Error loading Hugging Face model: {e}")

# 2. Gemini LLM (for interactive response generation)
# Check for API key and initialize client
GEMINI_API_KEY = "AIzaSyDFw11sdgNWwBRX2luLiXEJbNtQgfVk5Bo"
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini Client initialized.")
else:
    print("❌ Gemini API key not provided.")


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- Sentiment Classification Function (from previous steps) ---

# main.py (Inside classify_sentiment function)

def classify_sentiment(text):
    """
    Classifies sentiment using the RoBERTa model.
    Returns: ('POSITIVE'/'NEGATIVE'/'NEUTRAL', score)
    """
    if hf_sentiment_pipeline is None:
        return "NEUTRAL", 0.0

    result = hf_sentiment_pipeline(text)[0]
    hf_label = result['label']
    hf_score = result['score']
    
    # Map the RoBERTa model's numerical labels to your custom string constants
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }
    
    # Determine the final label using the map
    final_label = label_map.get(hf_label, "NEUTRAL")

    return final_label, hf_score # Return the correct final_label


# --- LLM Response Generation Function ---

def generate_llm_response(user_message, sentiment_label, sentiment_score):
    """
    Uses the Gemini LLM with a System Instruction to generate a natural response
    based on the classified sentiment.
    """
    if gemini_client is None:
        return "ERROR", "❌", "LLM service not available. Please set GEMINI_API_KEY."

    # Define the tone and personality of the chatbot (System Instruction)
    system_instruction = (
        "You are an empathetic and helpful Sentiment Chatbot. "
        "Your sole task is to generate a conversational response to the user's message. "
        "You MUST first acknowledge the user's classified sentiment. "
        "Sentiment Classification: {sentiment_label}. "
        "If the sentiment is POSITIVE, respond warmly. If NEGATIVE, respond with a sincere apology and offer help. If NEUTRAL, give a brief, helpful answer. "
        "DO NOT explicitly mention the sentiment score or the model you used. Just talk like a human."
    ).format(sentiment_label=sentiment_label)

    # Use a specific emoji based on sentiment (used in the response text)
    emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😟", "NEUTRAL": "😐"}
    emoji = emoji_map.get(sentiment_label, "❓")
    
    # We pass the full context to the LLM
    prompt = (
        f"The user wrote: '{user_message}'. "
        f"Please generate a response that matches the classified emotion."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7, # Add some creativity
            )
        )
        # The LLM generates the response, including the tone-matching
        return sentiment_label, emoji, response.text 
        
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return "ERROR", "❌", "Error communicating with the LLM. Check API key or network."


# --- Flask API Endpoint ---

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """ Handles the chat message request. """
    data = request.get_json(silent=True)
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Missing "message" key.'}), 400
        
    user_message = data['message']

    # 1. Classify Sentiment using Hugging Face (Fast ML)
    sentiment_label, sentiment_score = classify_sentiment(user_message)

    # 2. Generate Interactive Response using Gemini (LLM)
    final_label, sentiment_emoji, bot_response = generate_llm_response(
        user_message, 
        sentiment_label, 
        sentiment_score
    )

    # Return Final JSON Response
    return jsonify({
        'user_message': user_message,
        'sentiment': final_label,
        'sentiment_emoji': sentiment_emoji, 
        'chatbot_response': bot_response
    })

# Run the server locally
if __name__ == '__main__':
    print("🌐 Starting LLM-powered Flask server.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
