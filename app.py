#import torch
from flask import Flask, Response, render_template   
import cv2
import time


app = Flask(__name__)

#model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")

camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #results = model(frame)
            #frame = results.render()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            

@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)