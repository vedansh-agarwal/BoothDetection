from cv2 import threshold
from flask import Flask, jsonify, render_template, Response
import face_recognition as fr
from datetime import datetime
import numpy as np
import os.path
import json
import uuid
import cv2


baseloc = os.path.dirname(os.path.abspath(__file__))
fe_file = os.path.join(baseloc, "face_encodings.json")
config_file = os.path.join(baseloc, "config.json")

with open(config_file, 'r') as f:
    config = json.load(f)

threshold = config["threshold for face recognition"]
revisit_time = config["minimum time to count revisit in seconds"]

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        with open(fe_file, 'r') as f:
            face_emb = json.load(f)
    
        known_faces = list(face_emb.keys())
        known_face_encodings = [person_data[0] for person_data in face_emb.values()]

        success, img = camera.read()

        if not success:
            break

        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(imgS)
        face_encodings = fr.face_encodings(imgS, face_locations)

        for enc in face_encodings:
            faceDis = fr.face_distance(known_face_encodings, enc)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] <= threshold:
                time_since_visit = (datetime.now() - datetime.fromisoformat(face_emb[known_faces[matchIndex]][2])).total_seconds()
                if(time_since_visit >= revisit_time):
                    print("repeat visitor")
                    config["returning visitors"] += 1
                face_emb[known_faces[matchIndex]][2] = datetime.now().isoformat()
            else:
                print("new visitor")
                config["new visitors"] += 1
                face_emb[str(uuid.uuid4())] = [enc.tolist(), datetime.now().isoformat(), datetime.now().isoformat()]

            with open(config_file, 'w') as f:
                json.dump(config, f)

            with open(fe_file, 'w') as f:
                json.dump(face_emb, f)

        _, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_details')
def get_details():
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        response = jsonify(new_visitors=config["new visitors"], returning_visitors=config["returning visitors"])
    except:
        response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(debug=True)
