from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import datetime
import json
import joblib
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import pickle
import pandas as pd
import cv2
from collections import Counter
import nltk
import string
import whisper
import sounddevice as sd
import wave

app = Flask(__name__)
CORS(app)

# Google Fit API Configuration
SCOPES = [
    'https://www.googleapis.com/auth/fitness.heart_rate.read',
    'https://www.googleapis.com/auth/fitness.sleep.read'
]
DB_NAME = 'updated.db'

# Define explicit path to Haar cascade file
HAAR_CASCADE_PATH = "fmodels/haarcascade_frontalface_default.xml"

# Load Smartwatch Models
try:
    watch_model = joblib.load("wmodels/XGBoost_0.9275.pkl")
    watch_label_encoder = joblib.load("wmodels/label_encoder.pkl")
    scaler = joblib.load("wmodels/scaler.pkl")
except Exception as e:
    print(f"Error loading smartwatch models: {e}")
    watch_model = watch_label_encoder = scaler = None

# Load Facial Model with Minimal Logging
try:
    from tensorflow.keras.models import load_model
    facial_model = load_model("fmodels/fer_stress_model.h5")
    facial_label_encoder = np.load("fmodels/fer_label_classes.npy", allow_pickle=True)
except Exception as e:
    print(f"Failed to load TensorFlow or facial model: {str(e)}")
    import traceback
    traceback.print_exc()
    facial_model = None
    facial_label_encoder = None

# Load Voice Model
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    voice_model = joblib.load("smodels/bestmodel_Support Vector Machine.pkl")
    voice_vectorizer = joblib.load("smodels/tfidf_vectorizer.pkl")
    voice_label_mapping = {0: "Low", 1: "Neutral", 2: "High"}
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading voice models: {e}")
    voice_model = voice_vectorizer = voice_label_mapping = whisper_model = None

# Database Initialization
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS health_data (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                date TEXT,
                                sleep_duration_minutes INTEGER,
                                heart_rate_data TEXT,
                                avg_hr INTEGER,
                                min_hr INTEGER,
                                max_hr INTEGER,
                                hr_std_dev REAL
                            )''')
        
        # Create a new table to store stress detection results
        cursor.execute('''CREATE TABLE IF NOT EXISTS stress_results (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id TEXT,
                                face_stress TEXT,
                                voice_stress TEXT,
                                watch_stress TEXT,
                                combined_stress TEXT,
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                            )''')
        conn.commit()
        print("Database initialized successfully with health_data and stress_results tables")
    except sqlite3.Error as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

# Run database initialization when app starts
init_db()

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

# Generate a unique session ID
def generate_session_id():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# Store stress result in the database
def store_stress_result(session_id, method, stress_level):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Check if this session exists
        cursor.execute('''SELECT * FROM stress_results WHERE session_id = ?''', (session_id,))
        row = cursor.fetchone()
        
        if row:
            # Update existing session
            if method == 'face':
                cursor.execute('''UPDATE stress_results SET face_stress = ? WHERE session_id = ?''', (stress_level, session_id))
            elif method == 'voice':
                cursor.execute('''UPDATE stress_results SET voice_stress = ? WHERE session_id = ?''', (stress_level, session_id))
            elif method == 'watch':
                cursor.execute('''UPDATE stress_results SET watch_stress = ? WHERE session_id = ?''', (stress_level, session_id))
            
            # Recalculate combined stress
            cursor.execute('''SELECT face_stress, voice_stress, watch_stress FROM stress_results WHERE session_id = ?''', (session_id,))
            stresses = cursor.fetchone()
            combined_stress = calculate_combined_stress(stresses[0], stresses[1], stresses[2])
            cursor.execute('''UPDATE stress_results SET combined_stress = ? WHERE session_id = ?''', (combined_stress, session_id))
        else:
            # Create new session
            face_stress = stress_level if method == 'face' else None
            voice_stress = stress_level if method == 'voice' else None
            watch_stress = stress_level if method == 'watch' else None
            combined_stress = calculate_combined_stress(face_stress, voice_stress, watch_stress)
            
            cursor.execute('''INSERT INTO stress_results (session_id, face_stress, voice_stress, watch_stress, combined_stress)
                            VALUES (?, ?, ?, ?, ?)''', (session_id, face_stress, voice_stress, watch_stress, combined_stress))
        
        conn.commit()
        print(f"Stored {method} stress result '{stress_level}' for session {session_id}")
        
        # Get the updated combined result
        cursor.execute('''SELECT combined_stress FROM stress_results WHERE session_id = ?''', (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Database error in store_stress_result: {str(e)}")
        return None
    finally:
        conn.close()

# Calculate combined stress level based on individual results
def calculate_combined_stress(face_stress, voice_stress, watch_stress):
    # Count how many methods returned results
    valid_results = [s for s in [face_stress, voice_stress, watch_stress] if s is not None]
    if not valid_results:
        return None
    
    # Count high stress occurrences
    high_count = sum(1 for s in valid_results if "High" in s)
    
    # If more than half of the valid results indicate high stress, return high stress
    if high_count >= len(valid_results) / 2:
        return "High Stress"
    else:
        return "Low Stress"

# Get the latest session or create a new one
def get_or_create_session():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get the most recent session from the last hour
        one_hour_ago = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''SELECT session_id FROM stress_results WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 1''', (one_hour_ago,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return generate_session_id()
    except sqlite3.Error as e:
        print(f"Database error in get_or_create_session: {str(e)}")
        return generate_session_id()
    finally:
        conn.close()

# Google Fit Credentials
def get_credentials():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

# Smartwatch Data Functions
def get_heart_rate_data(service):
    data_source = "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=7)
    
    body = {
        "aggregateBy": [{"dataTypeName": "com.google.heart_rate.bpm", "dataSourceId": data_source}],
        "bucketByTime": {"durationMillis": 900000},
        "startTimeMillis": int(start_time.timestamp() * 1000),
        "endTimeMillis": int(end_time.timestamp() * 1000)
    }
    
    response = service.users().dataset().aggregate(userId="me", body=body).execute()
    heart_rate_data = []

    for bucket in response.get('bucket', []):
        timestamp = datetime.datetime.fromtimestamp(int(bucket.get('startTimeMillis')) / 1000)
        for dataset in bucket.get('dataset', []):
            for point in dataset.get('point', []):
                if point.get('value'):
                    heart_rate_data.append({'timestamp': timestamp, 'heart_rate_bpm': point['value'][0]['fpVal']})
    
    df = pd.DataFrame(heart_rate_data)
    
    if not df.empty:
        df['heart_rate_bpm_int'] = df['heart_rate_bpm'].astype(int)
        hr_list = df['heart_rate_bpm_int'].tolist()
        hr_list = hr_list[-24:] if len(hr_list) >= 24 else hr_list
        min_hr = int(min(hr_list)) if hr_list else 0
        max_hr = int(max(hr_list)) if hr_list else 0
        avg_hr = round(sum(hr_list) / len(hr_list)) if hr_list else 0
        std_hr = round(np.std(hr_list), 1) if hr_list else 0
    else:
        hr_list, avg_hr, min_hr, max_hr, std_hr = [], 0, 0, 0, 0

    return hr_list, avg_hr, min_hr, max_hr, std_hr

def get_sleep_data_past_day(service):
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=1)
    request = service.users().sessions().list(
        userId="me",
        startTime=start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        endTime=end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        activityType=72
    )
    response = request.execute()
    print("Sleep API response:", json.dumps(response, indent=2))
    sleep_data = []
    for session in response.get('session', []):
        start_millis = int(session['startTimeMillis'])
        end_millis = int(session['endTimeMillis'])
        duration_minutes = (end_millis - start_millis) / (1000 * 60)
        sleep_data.append({
            'start_time': datetime.datetime.fromtimestamp(start_millis / 1000),
            'end_time': datetime.datetime.fromtimestamp(end_millis / 1000),
            'duration_minutes': duration_minutes
        })
    if sleep_data:
        print(f"Found {len(sleep_data)} sleep sessions, last duration: {sleep_data[-1]['duration_minutes']} minutes")
        return round(sleep_data[-1]['duration_minutes'])
    print("No sleep sessions found in the last 24 hours")
    return 0

def store_health_data(sleep_duration, hr_list, avg_hr, min_hr, max_hr, std_hr):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        hr_json = json.dumps(hr_list)
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''SELECT id FROM health_data WHERE date = ?''', (current_date,))
        existing_entry = cursor.fetchone()
        if existing_entry:
            cursor.execute('''UPDATE health_data SET sleep_duration_minutes = ?, heart_rate_data = ?, avg_hr = ?, min_hr = ?, max_hr = ?, hr_std_dev = ? WHERE date = ?''',
                           (sleep_duration, hr_json, avg_hr, min_hr, max_hr, std_hr, current_date))
        else:
            cursor.execute('''INSERT INTO health_data (date, sleep_duration_minutes, heart_rate_data, avg_hr, min_hr, max_hr, hr_std_dev) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (current_date, sleep_duration, hr_json, avg_hr, min_hr, max_hr, std_hr))
        conn.commit()
        print(f"Stored health data for {current_date}: sleep_duration={sleep_duration}")
    except sqlite3.Error as e:
        print(f"Database error in store_health_data: {str(e)}")
        raise
    finally:
        conn.close()

# Preprocess Face for Facial Model
def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

# Preprocess Text for Voice Model
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    custom_stop_words = ["the", "a", "an", "is", "am", "are", "was", "were"]
    tokens = [word for word in tokens if word not in custom_stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Get current session status
@app.route('/get-session-status', methods=['GET'])
def get_session_status():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
            
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''SELECT face_stress, voice_stress, watch_stress, combined_stress FROM stress_results WHERE session_id = ?''', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"error": "Session not found"}), 404
            
        return jsonify({
            "face_stress": result[0],
            "voice_stress": result[1],
            "watch_stress": result[2],
            "combined_stress": result[3],
            "completed": all(result[0:3])
        })
    except Exception as e:
        print(f"Error in get_session_status: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Analyze Video for Facial Stress with Debugging Logs
@app.route('/analyze/face', methods=['POST'])
def analyze_face_video():
    try:
        print("Received request to /analyze/face")
        if 'video' not in request.files:
            print("Error: No video file provided")
            return jsonify({"error": "No video file provided"}), 400
        
        if facial_model is None or facial_label_encoder is None:
            print("Error: Facial model not loaded")
            return jsonify({"error": "Facial model not loaded due to TensorFlow issue"}), 500
        
        video_file = request.files['video']
        video_path = os.path.join("temp", "uploaded_video.webm")
        os.makedirs("temp", exist_ok=True)
        video_file.save(video_path)
        if not os.path.exists(video_path):
            print("Error: Failed to save video file")
            return jsonify({"error": "Failed to save video file"}), 500
        print("Video saved to:", video_path)

        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            print(f"Error: Failed to load Haar cascade from {HAAR_CASCADE_PATH}")
            return jsonify({"error": f"Failed to load face cascade from {HAAR_CASCADE_PATH}"}), 500

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Failed to open video file")
            return jsonify({"error": "Failed to open video file"}), 500
        print("Video file opened successfully")

        emotion_list = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                processed_face = preprocess_face(face)
                prediction = facial_model.predict(processed_face, verbose=0)
                predicted_label = np.argmax(prediction)
                emotion = facial_label_encoder[predicted_label]
                emotion_list.append(emotion)

        cap.release()
        os.remove(video_path)
        print(f"Processed {frame_count} frames, found {len(emotion_list)} emotions")

        if not emotion_list:
            print("Error: No faces detected in video")
            return jsonify({"error": "No faces detected in video"}), 400
        
        most_common_emotion = Counter(emotion_list).most_common(1)[0][0]
        print(f"Most common emotion: {most_common_emotion}")
        
        # Get or create a session and store the result
        session_id = request.form.get('session_id', get_or_create_session())
        combined_stress = store_stress_result(session_id, 'face', most_common_emotion)
        
        return jsonify({
            "session_id": session_id,
            "stress_level": most_common_emotion,
            "combined_stress": combined_stress
        })
    
    except Exception as e:
        print(f"Unexpected error in analyze_face_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Analyze Voice Stress with Debugging Logs
@app.route('/analyze_voice_stress', methods=['POST'])
def analyze_voice_stress():
    try:
        print("Received request to /analyze_voice_stress")
        
        if voice_model is None or voice_vectorizer is None or whisper_model is None:
            print("Error: Voice model not loaded")
            return jsonify({"error": "Voice model not loaded due to model issue"}), 500
        
        # Check if the request has form data or direct audio blob
        if request.files and 'audio' in request.files:
            print("Processing audio from form data")
            audio_file = request.files['audio']
            audio_path = os.path.join("temp", "uploaded_audio.webm")
            audio_file.save(audio_path)
            # Extract session_id if provided in form data
            session_id = request.form.get('session_id', get_or_create_session())
        else:
            print("Processing audio from direct blob")
            # Handle direct audio blob
            audio_path = os.path.join("temp", "uploaded_audio.webm")
            with open(audio_path, 'wb') as f:
                f.write(request.data)
            # Get session_id from JSON data if available
            try:
                json_data = request.get_json(silent=True)
                session_id = json_data.get('session_id') if json_data else get_or_create_session()
            except:
                session_id = get_or_create_session()
        
        if not os.path.exists(audio_path):
            print("Error: Failed to save audio file")
            return jsonify({"error": "Failed to save audio file"}), 500
        
        print("Audio saved to:", audio_path)
        print("Audio file size:", os.path.getsize(audio_path))

        # Process the audio with Whisper
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        print(f"Transcription: {transcription}")
        
        if not transcription.strip():
            os.remove(audio_path)
            return jsonify({"error": "No speech detected in the audio"}), 400
        
        processed_text = preprocess_text(transcription)
        text_features = voice_vectorizer.transform([processed_text])
        prediction = voice_model.predict(text_features)[0]
        stress_level = voice_label_mapping[prediction]
        print(f"Predicted stress level: {stress_level}")
        
        os.remove(audio_path)
        
        # Store the result and get combined stress
        combined_stress = store_stress_result(session_id, 'voice', stress_level)
        
        return jsonify({
            "session_id": session_id,
            "transcribed_text": transcription,
            "stress_level": stress_level,
            "combined_stress": combined_stress
        })
    
    except Exception as e:
        print(f"Unexpected error in analyze_voice_stress: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Fetch Data and Analyze Smartwatch Data
@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    try:
        session_id = request.args.get('session_id', get_or_create_session())
        
        creds = get_credentials()
        service = build('fitness', 'v1', credentials=creds)
        hr_list, avg_hr, min_hr, max_hr, std_hr = get_heart_rate_data(service)
        sleep_duration = get_sleep_data_past_day(service)
        store_health_data(sleep_duration, hr_list, avg_hr, min_hr, max_hr, std_hr)
        return jsonify({
            "session_id": session_id,
            "message": "Data fetched and stored successfully"
        })
    except Exception as e:
        print(f"Error fetching smartwatch data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500
    


# Predict Stress with Smartwatch Data
@app.route('/predict-stress', methods=['GET'])
def predict_stress():
    if not watch_model:
        return jsonify({"error": "Smartwatch model not loaded"}), 500
    
    try:
        session_id = request.args.get('session_id', get_or_create_session())
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''SELECT sleep_duration_minutes, heart_rate_data, avg_hr, min_hr, max_hr, hr_std_dev FROM health_data WHERE date = ?''', (current_date,))
        data = cursor.fetchone()
        conn.close()
        
        if data:
            sleep_duration, hr_json, avg_hr, min_hr, max_hr, std_hr = data
            hr_list = json.loads(hr_json)
            input_data = np.array([sleep_duration] + hr_list + [avg_hr, min_hr, max_hr, std_hr]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            prediction = watch_model.predict(input_scaled)[0]
            predicted_class = watch_label_encoder.inverse_transform([prediction])[0]
            
            # Store the result
            combined_stress = store_stress_result(session_id, 'watch', predicted_class)
            
            return jsonify({
                "session_id": session_id,
                "stress_level": predicted_class,
                "combined_stress": combined_stress
            })
        else:
            return jsonify({"error": "No data found for today"}), 400
    except sqlite3.Error as e:
        print(f"Database error in predict_stress: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
