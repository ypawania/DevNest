import torch
from flask import Flask, Response, render_template, jsonify
import cv2
import time
import numpy as np
import warnings
from util import get_response

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")

camera = cv2.VideoCapture(0)

detected_animals = {}

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            results = model(frame)
            filtered_results = results.xyxy[0][results.xyxy[0][:, 4] >= 0.1]

            for *box, conf, cls in filtered_results:
                animal = model.names[int(cls)]
                if animal not in detected_animals:
                    summary = get_response(animal)
                    detected_animals[animal] = summary

                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            

@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html", detected_animals=detected_animals)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/summaries")
def summaries():
    return jsonify(detected_animals)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)