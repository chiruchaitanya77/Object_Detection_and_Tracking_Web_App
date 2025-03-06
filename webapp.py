import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            #os.makedirs('runs/detect', exist_ok=True)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)

                # Perform the detection
                model = YOLO('yolov9c.pt')
                detections = model(img, save=True)
                return display(f.filename)

            elif file_extension == 'mp4':
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec if needed (e.g., 'XVID')
                fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS to maintain consistency
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                model = YOLO('yolov9c.pt')
                num_frames_to_process = 5 * fps
                # 5 * fps takes 37 seconds to
                # Process each frame in the video
                frame_count = 0
                ret, frame = cap.read()
                while cap.isOpened():
                    if (num_frames_to_process > 0 and frame_count >= num_frames_to_process) or not ret:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break  # Exit loop if there are no frames left to process

                    # Perform detection on the current frame
                    results = model(frame,save=True)

                    # Annotate the frame with detected objects
                    annotated_frame = results[0].plot()  # 'plot()' adds bounding boxes and labels to the frame

                    # Write the processed frame to the output video
                    out.write(annotated_frame)

                    frame_count += 1
                 # Release the video capture object and close all OpenCV windows
                cap.release()
                out.release()
                # Closes all the frames
                cv2.destroyAllWindows()
                return video_feed()



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    if not os.path.exists(folder_path):
        return "Error: The folder 'runs/detect' does not exist."

    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return "Error: No subfolders found in 'runs/detect'."

    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)
    files = os.listdir(directory)
    if not files:
        return "Error: No files found in the latest subfolder."

    latest_file = files[0]

    print("Latest file: ", latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        # return render_template('index.html', image_path='run/detect')
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)  #control the frame rate to display one frame every 100 milliseconds:


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
    # return render_template('index.html', video_path='run/detect')


def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print(type(frame))

            img = Image.open(io.BytesIO(frame))

            model = YOLO('yolov9c.pt')
            results = model(img, save=True)

            print(results)
            cv2.waitKey(1)

            annotated_frame = results[0].plot()  # 'plot()' adds bounding boxes and labels to the frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            img_BGR = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5050, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('yolov9c.pt')
    app.run(port=5050)
