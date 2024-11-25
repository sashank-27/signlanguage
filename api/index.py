from flask import Flask, Response, render_template, send_from_directory
import cv2
import torch
from ultralytics import YOLO
import os
from http.server import BaseHTTPRequestHandler

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO('./best.pt')

def generate_frames():
    # Your existing generate_frames code here
    pass

@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# Handler for Vercel
def handler(request):
    with app.request_context(request):
        return app.handle_request()
