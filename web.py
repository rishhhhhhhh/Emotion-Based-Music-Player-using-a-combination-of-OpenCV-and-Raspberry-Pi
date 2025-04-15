from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import cv2
import os
import numpy as np
import tensorflow as tf
import pygame
from mutagen.mp3 import MP3
import time
from PIL import Image

app = Flask(__name__)
pygame.mixer.init()

model = tf.keras.models.load_model('emotion_detection_model.h5')

CLASS_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
MUSIC_DB = {
    "happy": "static/music_database/enhance/happy/",
    "sad": "static/music_database/enhance/sad/",
    "angry": "static/music_database/enhance/angry/",
    "neutral": "static/music_database/enhance/neutral/",
    "fear": "static/music_database/enhance/fear/",
    "disgust": "static/music_database/enhance/disgust/",
    "surprise": "static/music_database/enhance/surprise/"
}
DEFAULT_FOLDER = "static/music_database/enhance/neutral/"
current_playlist = []
current_index = 0
is_paused = False
current_volume = 0.5

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def resize_image(image_path, size=(148, 148)):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize(size, Image.LANCZOS)
        img.save(image_path)

def detect_emotion():
    try:
        image_path = "static/user.jpg"
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)  # Flip horizontally
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            print("❌ No face detected. Using default emotion: Neutral")
            return "neutral"

        x, y, w, h = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        print("🧠 Face shape before resize:", face_crop.shape)

        face_crop = cv2.resize(face_crop, (48, 48))
        print("🧠 Face shape after resize:", face_crop.shape)

        face_crop = face_crop / 255.0
        face_crop = np.reshape(face_crop, (1, 48, 48, 1))
        print("🧠 Face shape after reshape:", face_crop.shape)

        prediction = model.predict(face_crop)
        print("🔮 Prediction vector:", prediction)

        predicted_index = np.argmax(prediction)
        detected_emotion = CLASS_LABELS[predicted_index]
        print("🎯 Predicted emotion:", detected_emotion)

        return detected_emotion if detected_emotion in MUSIC_DB else "neutral"

    except Exception as e:
        print(f"⚠️ Error in emotion detection: {e}")
        return "neutral"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    print("📹 Starting video stream...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    time.sleep(1)

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read video frame")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/capture', methods=['POST'])
def capture():
    print("🎥 Attempting to open webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    time.sleep(1)

    success = False
    frame = None

    for i in range(5):
        success, frame = cap.read()
        if success:
            print(f"✅ Frame captured on attempt {i+1}")
            break
        else:
            print(f"⚠️ Retrying webcam capture (attempt {i+1})")
            time.sleep(0.5)

    cap.release()

    if not success:
        print("❌ All retries failed.")
        return render_template('emotion.html', emotion="neutral", image_url="")

    cv2.imwrite("static/user.jpg", frame)
    print("📸 Image saved.")

    emotion = detect_emotion()
    image_url = f"static/user.jpg?t={int(time.time())}"
    return render_template('emotion.html', emotion=emotion, image_url=image_url)

@app.route('/music/<mood>')
def music(mood):
    global current_playlist, current_index

    folder = MUSIC_DB.get(mood, DEFAULT_FOLDER)

    if os.path.exists(folder):
        current_playlist = [os.path.join(folder, song) for song in os.listdir(folder) if song.endswith(".mp3")]
        current_index = 0

    if current_playlist:
        song_path = current_playlist[current_index]
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(current_volume)
        duration = int(MP3(song_path).info.length)

        song_name = os.path.basename(song_path).replace(".mp3", "")
        album_image = f"static/album_images/{song_name}.jpg"
        if not os.path.exists(album_image):
            album_image = "static/album_images/default.jpg"
        else:
            resize_image(album_image)
    else:
        song_path, duration, album_image = None, 0, "static/album_images/default.jpg"

    return render_template('music.html', mood=mood, song=song_path, duration=duration, album_image=album_image)

@app.route('/next_song', methods=['POST'])
def next_song():
    global current_playlist, current_index

    if current_playlist:
        current_index = (current_index + 1) % len(current_playlist)
        song_path = current_playlist[current_index]
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(current_volume)
        duration = int(MP3(song_path).info.length)

        song_name = os.path.basename(song_path).replace(".mp3", "")
        album_image = f"static/album_images/{song_name}.jpg"
        if not os.path.exists(album_image):
            album_image = "static/album_images/default.jpg"
        else:
            resize_image(album_image)
    else:
        song_path, duration, album_image = None, 0, "static/album_images/default.jpg"

    return jsonify({"song": song_path, "duration": duration, "album_image": album_image})

@app.route('/prev_song', methods=['POST'])
def prev_song():
    global current_playlist, current_index

    if current_playlist:
        current_index = (current_index - 1) % len(current_playlist)
        song_path = current_playlist[current_index]
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(current_volume)
        duration = int(MP3(song_path).info.length)

        song_name = os.path.basename(song_path).replace(".mp3", "")
        album_image = f"static/album_images/{song_name}.jpg"
        if not os.path.exists(album_image):
            album_image = "static/album_images/default.jpg"
        else:
            resize_image(album_image)
    else:
        song_path, duration, album_image = None, 0, "static/album_images/default.jpg"

    return jsonify({"song": song_path, "duration": duration, "album_image": album_image})

@app.route('/toggle_play', methods=['POST'])
def toggle_play():
    global is_paused

    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False
        return jsonify({"status": "playing"})
    else:
        pygame.mixer.music.pause()
        is_paused = True
        return jsonify({"status": "paused"})

@app.route('/set_volume', methods=['POST'])
def set_volume():
    global current_volume
    volume = float(request.form.get('volume', 0.5))
    pygame.mixer.music.set_volume(volume)
    current_volume = volume
    return jsonify({"volume": volume})

@app.route('/stop', methods=['POST'])
def stop_music():
    pygame.mixer.music.stop()
    return redirect(url_for('home'))

@app.route('/select_mood')
def select_mood():
    current_mood = request.args.get('current_mood', 'neutral')
    return render_template('select_mood.html', current_mood=current_mood)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
