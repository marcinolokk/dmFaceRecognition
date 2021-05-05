import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

imgsPath = 'data/mixedPhotos/'
destPath = 'data/mixedFaces/'   
face_counter = 0

def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames

def getFilesNames(imgsPath):
    return [f for f in listdir(imgsPath) if isfile(join(imgsPath, f))]

def readImg(path):
    # Load image
    img_path = path
    image = io.imread(img_path)
    return image

# Detect faces
def detectFaces(image):
    detected_faces = detect_faces(image)
    return detected_faces

# Crop faces and plot
def cropFaces(detected_faces):
    for n, face_rect in enumerate(detected_faces):
        global face_counter
        face_counter += 1

        face = Image.fromarray(image).crop(face_rect)
        # plt.subplot(1, len(detected_faces), n+1)
        # plt.imshow(face)
        plt.axis('off')
        plt.imshow(face)
        plt.savefig(destPath + str(face_counter) + '.png')


for filename in getFilesNames(imgsPath):
    path = imgsPath + filename
    # print(path)
    # each of the paths
    image = readImg(path)
    detected_faces = detectFaces(image)
    cropFaces(detected_faces)