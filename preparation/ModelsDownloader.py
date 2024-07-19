import gdown
import os
from LaserFR.deepface.commons import functions
import zipfile
import requests


def VGGModel(url='https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'):
    # -----------------------------------

    home = functions.get_deepface_home()
    output = home + '/.deepface/weights/vgg_face_weights.h5'

    if not os.path.isfile(output):
        print("vgg_face_weights.h5 will be downloaded...")
        gdown.download(url, output, quiet=False)


def DeepFaceModel(
        url='https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned'
            '/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'):
    # ---------------------------------

    home = functions.get_deepface_home()

    if not os.path.isfile(home + '/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5'):
        print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")

        output = home + '/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'

        gdown.download(url, output, quiet=False)

        # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(home + '/.deepface/weights/')


def ArcFaceModel(url='https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'):
    home = functions.get_deepface_home()

    file_name = "arcface_weights.h5"
    output = home + '/.deepface/weights/' + file_name

    if not os.path.isfile(output):
        print(file_name, " will be downloaded to ", output)
        gdown.download(url, output, quiet=False)


def SFaceModel(
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface"
            "/face_recognition_sface_2021dec.onnx"):
    home = functions.get_deepface_home()

    file_name = home + '/.deepface/weights/face_recognition_sface_2021dec.onnx'

    if not os.path.isfile(file_name):
        print("sface weights will be downloaded...")

        gdown.download(url, file_name, quiet=False)


ArcFaceModel()
VGGModel()
DeepFaceModel()
SFaceModel()
