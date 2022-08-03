from flask import Flask, jsonify, render_template, Response
from datetime import datetime
import face_recognition as fr
import numpy as np
import logging
import os.path
import json
import uuid
import cv2


baseloc = os.path.dirname(os.path.abspath(__file__))
fe_file = os.path.join(baseloc, "face_encodings.json")
config_file = os.path.join(baseloc, "config.json")
output_file = os.path.join(baseloc, "output.json")

with open(config_file, 'r') as f:
    config = json.load(f)

with open(output_file, 'r') as f:
    output = json.load(f)

threshold = config["threshold for face recognition"]
revisit_time = config["minimum time to count revisit in seconds"]
resize_factor = config["resize factor"]

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, img = camera.read()

        if not success:
            break

        imgS = cv2.resize(img, (0,0), None, resize_factor, resize_factor)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(imgS)
        face_encodings = fr.face_encodings(imgS, face_locations)

        try:
            with open(fe_file, 'r') as f:
                face_emb = json.load(f)
        except:
            face_emb = {}
            addFirstVisitor(face_emb, face_encodings[0])

        known_faces = list(face_emb.keys())
        known_face_encodings = [person_data[0] for person_data in face_emb.values()]

        for enc in face_encodings:
            faceDis = fr.face_distance(known_face_encodings, enc)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] <= threshold:
                time_since_visit = (datetime.now() - datetime.fromisoformat(face_emb[known_faces[matchIndex]][2])).total_seconds()
                if(time_since_visit >= revisit_time):
                    print("repeat visitor")
                    output["returning visitors"] += 1
                face_emb[known_faces[matchIndex]][2] = datetime.now().isoformat()
            else:
                print("new visitor")
                output["new visitors"] += 1
                face_emb[str(uuid.uuid4())] = [enc.tolist(), datetime.now().isoformat(), datetime.now().isoformat()]

            with open(output_file, 'w') as f:
                json.dump(output, f)

            with open(fe_file, 'w') as f:
                json.dump(face_emb, f)

        _, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def addFirstVisitor(face_emb, enc):
    print("new visitor")
    output["new visitors"] += 1
    face_emb[str(uuid.uuid4())] = [enc.tolist(), datetime.now().isoformat(), datetime.now().isoformat()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_details')
def get_details():
    try:
        with open(output_file, 'r') as f:
            output = json.load(f)
        response = jsonify(new_visitors=output["new visitors"], returning_visitors=output["returning visitors"])
    except:
        response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(debug=True)
