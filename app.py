from flask import Flask, render_template, request, jsonify
from medical.data_ingestion import data_ingestion
from medical.retrieval_generation import generation
import pyttsx3
import speech_recognition as sr
import whisper
import logging
from gtts import gTTS
from pydub import AudioSegment
import os
from pydub.utils import which
#from googletrans import Translator, LANGUAGES  # For translation
from deep_translator import GoogleTranslator  # For translation
#AudioSegment.converter ="C:/Users/satis/OneDrive/Desktop/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"
# Initialize data ingestion, OpenAI API, and TTS engine
vstore = data_ingestion("done")
logging.basicConfig(level=logging.DEBUG)
whisper_model = whisper.load_model("base")
app = Flask(__name__)
chain = generation(vstore)
tts_engine = pyttsx3.init()
# Translator instance
translator = GoogleTranslator()
os.environ["WHISPER_CACHE"] = "/app/.cache"  # Adjust this to your desired path
LANGUAGES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    # Add more languages as needed
}
# Default language
default_language = "en"
# Translation function
def translate_text(text, target_lang):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return text  # Return original text on error


# Set TTS parameters
tts_engine.setProperty('rate', 150)  # Speed
tts_engine.setProperty('volume', 1)  # Volume

@app.route("/")
def index():
    return render_template("index.html",languages=LANGUAGES)

@app.route("/get", methods=["POST", "GET"])
def chat():
   
   if request.method == "POST":
      msg = request.form["msg"]
      selected_lang = request.form.get("language", default_language)
      input = msg

      result = chain.invoke(
         {"input": input},
    config={
        "configurable": {"session_id": "satish"}
    },
)["answer"]
      print(result)
      print(selected_lang)
       # Translate response if needed
      if selected_lang != "en":
            translated_result = translate_text(result, target_lang=selected_lang)
      else:
            translated_result = result
      print(translated_result)
     

      return str(translated_result)

@app.route("/voice", methods=["POST"])
def voice():
    recognizer = sr.Recognizer()
    audio_file = request.files.get("audio", None)
    selected_lang = request.form.get("language", default_language)


    if not audio_file:
        logging.error("No audio received")
        return jsonify({"error": "No audio received"}), 400

    try:
        # Save the audio temporarily
        audio_path = "static/temp_audio.webm"
        audio_file.save(audio_path)

        # Convert audio to WAV using pydub
        wav_path = "static/temp_audio.wav"
        AudioSegment.from_file(audio_path).export(wav_path, format="wav")

        # Process the WAV file with speech_recognition
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

        # Get chatbot response
        session_id = "default_session"
        response = chain.invoke(
            {"input": text},
            config={"configurable": {"session_id": session_id}}
        )["answer"]
        print(selected_lang)
        if selected_lang != "en":
            translated_result = translate_text(response, target_lang=selected_lang)
            text=translate_text(text, target_lang=selected_lang)
        else:
            translated_result = response
        print(translated_result)
        # Convert response to speech
        tts = gTTS(text=translated_result, lang=selected_lang, slow=False)
        #tts_engine.runAndWait()
        tts.save("static/response.mp3")
        
        return jsonify({
            "userMessage": text,  # Add the transcription
            "response": translated_result,
            "audio": "/static/response.mp3"
        })
    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
        app.run(host="0.0.0.0", port=7860)