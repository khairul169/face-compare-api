from flask import Flask, request
from face_recognition import FaceRecognition
import response as res
import os
import json
import numpy

face = FaceRecognition()
app = Flask(__name__)

@app.get('/')
def main():
    return res.success({"hello": "world!"})

def get_file_descriptor(file):
    path = os.path.join('./tmp/', file.filename)
    file.save(path)

    try:
        descriptor = face.get_face_descriptor(path)
        return descriptor
    except:
        raise Exception("Err processing %s!" % (file.filename))
    finally:
        os.remove(path)

@app.post('/get-descriptor')
def get_face_descriptor():
    image_list = request.files.getlist('image')

    if not image_list:
        raise Exception("Please upload atleast one image!")
    
    result = []
    
    for image in image_list:
        descriptor = get_file_descriptor(image)
        
        if descriptor is False:
            raise Exception("Image doesn't contain any faces: %s" % (image.filename))
        
        result.append({
            "filename": image.filename,
            "descriptor": json.dumps(descriptor.tolist()),
        })
    
    return res.success(result)

@app.post('/compare')
def compare_faces():
    image = request.files.get('image')
    main_descriptor = request.form.get('main')

    if not image and not main_descriptor:
        raise Exception("Please specify image or main descriptor!")

    faces = request.form.getlist('faces')

    if not faces:
        raise Exception("Please specify atleast one face descriptor.")
    
    if image:
        main_descriptor = get_file_descriptor(image)
    else:
        main_descriptor = numpy.array(json.loads(main_descriptor))

    if descriptor is False:
        raise Exception("Main image doesn't contain any faces!")

    diffs = []

    for face_descriptor in faces:
        descriptor = numpy.array(json.loads(face_descriptor))
        diffs.append(face.compare_face(main_descriptor, descriptor))
    
    face_distance = numpy.mean(diffs)

    return res.success({
        "is_same": True if face_distance >= 0.4 else False,
        "distance": face_distance
    })

@app.errorhandler(Exception)
def error_handler(e: Exception):
    return res.error(str(e))

