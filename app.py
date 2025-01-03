from flask import Flask, Response, render_template, send_from_directory
import cv2
import torch
import time
import platform
from ultralytics import YOLO
from gtts import gTTS
import os
import threading

app = Flask(__name__)

# Load the YOLO model
model = YOLO('./best.pt')

CONFIDENCE_THRESHOLD = 0.5

# Detect platform to decide text-to-speech engine
is_mac = platform.system() == "Darwin"
if is_mac:
    print("macOS detected, skipping pyttsx3 and using gTTS instead")
    engine = None
else:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  
        engine.setProperty('volume', 0.9)
    except Exception as e:
        print(f"Error initializing pyttsx3, switching to gTTS: {e}")
        engine = None
# Flag to control ongoing speech
is_speaking = False

# Function to play audio using gTTS in a separate thread
def speak_with_gtts(text):
    global is_speaking
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("detected.mp3")
        print("gTTS: Audio saved as detected.mp3")
        
        # Play audio in the background using 'afplay' for macOS
        os.system("afplay detected.mp3")
        print("gTTS: Audio played using afplay")
    except Exception as e:
        print(f"Error using gTTS: {e}")
    finally:
        is_speaking = False  # Reset speaking flag when done

# Function to handle speaking in a separate thread
def speak_text(text):
    global is_speaking
    if not is_speaking:
        is_speaking = True
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error using pyttsx3: {e}")
                # Fallback to gTTS if pyttsx3 fails
                threading.Thread(target=speak_with_gtts, args=(text,)).start()
        else:
            # Use gTTS directly if pyttsx3 is unavailable
            threading.Thread(target=speak_with_gtts, args=(text,)).start()

def generate_frames():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    object_detected = False
    current_label = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        results = model(frame_resized)

        current_detection = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                current_detection = True

                label = model.names[cls]
                confidence = f'{conf:.2f}'

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f'{label}: {confidence}'

                # Change label text color to red
                cv2.putText(frame_resized, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if not object_detected:
                    speak_text(f"{label}")
                    current_label = label

        if object_detected and not current_detection:
            print("No object detected anymore, resetting speech.")
            is_speaking = False
            current_label = ""

        object_detected = current_detection

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(frame_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame = buffer.tobytes()

        # Yield both the frame and the current label
        yield (b'--frame\r\n'
               b'X-Detected-Label: ' + current_label.encode() + b'\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Static file handling (CSS, JS)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
