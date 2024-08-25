import torch
from flask import Flask, request, render_template   
import cv2
import time


app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")

camera = cv2.VideoCapture(0)
while not camera.isOpened():    
    time.sleep(0.1)

if __name__ == "__main__":
    app.run(use_reloader=False)