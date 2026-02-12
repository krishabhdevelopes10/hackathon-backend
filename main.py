from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import librosa
import numpy as np
from datetime import datetime
import json
import os

app = FastAPI()

# Allow Wix to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained AI models (this happens when server starts)
print("🤖 Loading AI models... please wait...")

# Model 1: Converts speech to text
transcriber = pipeline("automatic-speech-recognition", 
                      model="openai/whisper-tiny")  # Using tiny for faster loading

# Model 2: Detects emotions from voice
emotion_classifier = pipeline("audio-classification",
                             model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

print("✅ Models loaded! Backend ready!")

def analyze_speech_patterns(audio_path):
    """Analyzes audio for stress indicators"""
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Count pauses in speech (sign of hesitation/memory issues)
    intervals = librosa.effects.split(y, top_db=20)
    num_pauses = len(intervals) - 1
    
    # Measure pitch variation (monotone = low variation)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_std = np.std(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
    
    # Measure energy (fatigue = low energy)
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = np.mean(rms)
    
    # Speaking rate
    duration = len(y) / sr
    
    return {
        "num_pauses": int(num_pauses),
        "pitch_variation": float(pitch_std),
        "energy_level": float(avg_energy),
        "duration_seconds": float(duration)
    }

@app.get("/")
async def home():
    return {"message": "Backend is running! ✅"}

@app.post("/analyze-speech")
async def analyze_speech(user_id: str, audio: UploadFile = File(...)):
    """Main endpoint that receives audio and returns analysis"""
    
    print(f"📝 Received audio from user: {user_id}")
    
    # Save the uploaded audio temporarily
    audio_path = f"temp_{user_id}.wav"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # STEP 1: Convert speech to text using Whisper model
    print("🎤 Transcribing speech...")
    transcription = transcriber(audio_path)
    transcript_text = transcription["text"]
    
    # STEP 2: Detect emotions using emotion model
    print("😊 Analyzing emotions...")
    emotions = emotion_classifier(audio_path)
    
    # STEP 3: Analyze speech patterns
    print("📊 Analyzing speech patterns...")
    audio_features = analyze_speech_patterns(audio_path)
    
    # CALCULATE STRESS SCORE
    # Based on negative emotions
    emotion_stress = 0
    for emotion in emotions:
        if emotion['label'] in ['angry', 'sad', 'fear']:
            emotion_stress += emotion['score'] * 100
    
    # Based on pauses
    pause_factor = min(audio_features['num_pauses'] * 5, 40)
    
    # Based on monotone speech
    monotone_factor = 30 if audio_features['pitch_variation'] < 50 else 0
    
    # Total stress score (0-100)
    stress_score = min(emotion_stress + pause_factor + monotone_factor, 100)
    
    # DETECT WARNING SIGNS
    warnings = []
    if audio_features['num_pauses'] > 15:
        warnings.append("⚠️ Frequent pauses detected")
    if audio_features['pitch_variation'] < 50:
        warnings.append("⚠️ Monotone speech pattern")
    if len(transcript_text.split()) < 30:
        warnings.append("⚠️ Reduced verbal output")
    if stress_score > 70:
        warnings.append("⚠️ High stress levels")
    
    # Create analysis result
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript_text,
        "word_count": len(transcript_text.split()),
        "emotions": emotions[:3],  # Top 3 emotions
        "stress_score": round(stress_score, 1),
        "speech_rate": round(len(transcript_text.split()) / audio_features['duration_seconds'] * 60, 1),
        "warnings": warnings
    }
    
    # Save to file (simple database)
    data_file = "user_data.json"
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    if user_id not in data:
        data[user_id] = []
    
    data[user_id].append(analysis)
    
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Clean up temp file
    os.remove(audio_path)
    
    print(f"✅ Analysis complete! Stress score: {stress_score}")
    
    return {
        "status": "success",
        "analysis": analysis
    }

@app.get("/get-trends/{user_id}")
async def get_trends(user_id: str):
    """Get user's history and trends"""
    
    data_file = "user_data.json"
    if not os.path.exists(data_file):
        return {"status": "no_data", "message": "No recordings yet"}
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if user_id not in data:
        return {"status": "no_data", "message": "No recordings for this user"}
    
    user_recordings = data[user_id]
    
    # Calculate averages
    recent = user_recordings[-5:]  # Last 5 recordings
    avg_stress = sum(r["stress_score"] for r in recent) / len(recent)
    avg_speech_rate = sum(r["speech_rate"] for r in recent) / len(recent)
    
    # Trend detection
    if len(user_recordings) >= 3:
        recent_stress = [r["stress_score"] for r in user_recordings[-3:]]
        if all(recent_stress[i] < recent_stress[i+1] for i in range(len(recent_stress)-1)):
            trend = "increasing"
        elif all(recent_stress[i] > recent_stress[i+1] for i in range(len(recent_stress)-1)):
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "total_recordings": len(user_recordings),
        "average_stress": round(avg_stress, 1),
        "average_speech_rate": round(avg_speech_rate, 1),
        "trend": trend,
        "recent_recordings": recent,
        "all_recordings": user_recordings
    }
