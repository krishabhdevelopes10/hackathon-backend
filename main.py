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

# Models disabled for demo - using mock data
print("✅ Backend ready! (Using mock data for testing)")

@app.get("/")
async def home():
    return {"message": "Backend is running! ✅"}

@app.post("/analyze-speech")
async def analyze_speech(user_id: str, audio: UploadFile = File(...)):
    """Main endpoint that receives audio and returns mock analysis"""
    
    print(f"📝 Received audio from user: {user_id}")
    await audio.read()  # Read and discard the audio for now
    
    # Mock analysis data
    import random
    stress_score = random.randint(20, 80)
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "transcript": "This is a test recording. The backend is working correctly and ready for integration.",
        "word_count": 13,
        "emotions": [
            {"label": "calm", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "happy", "score": 0.1}
        ],
        "stress_score": round(stress_score, 1),
        "speech_rate": 120.0,
        "warnings": ["⚠️ Test data - no actual audio analysis"] if stress_score > 70 else []
    }
    
    # Save to file
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
    
    print(f"✅ Mock analysis complete! Stress score: {stress_score}")
    
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
