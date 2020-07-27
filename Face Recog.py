# There are three types of code for face recognition
# Code 1 - Trainer(it takes 30 pictures of you(number can be changes) And crops the face part and save it)
# Code 2 - Aggregator(Creates folder (dataset) and puts all 30 pics in that and assigns IDs for each folder
# Code 3 - Recogniser(Takes data from dataset and real time webcam and tells which face is whose)
# write all the codes one by one and keep removing it after its execution

# This is trainer code down there
import cv2
import os
# import os = (operating system)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
#This is for set the width and height of the WebCam. this function has two arguments
# enumeration and pixels/enumeration_value. if enumeration is
# 0 (use in CV_CAP_PROS_POSMSEC),
# 1 = (THIS MEANS CV_CAP_PROS_POS_FRAMES)
# 2 = (use is CV_CAP_PROS_POS_AVI_RATIO),
# 3 = CV_CAP_PROS_POS_FRAME_WIDTH,
# 4 = CV_CAP_PROS_POS_FRAME_HEIGHT,
# 5 = CV_CAP_PROS_POS_FPS
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# haarcascade stores:-1. Edge feature, line feature and four rectangle feature
# It returns 1 if all stages of cascade features are passed successfully otherwise returns '0'
# here we store the data which decides whether it is a face or not(this file needs to be downloades and put in the same location os our python program)
# there are three types of cascade Edge cascade(takes combination of [D,B] or [B,D],
# Line cascade(takes [DBD]or [BDB] and 4 rectangles cascade(takes D/B in a format of 2x2 identity matrix)
# D=Dark spot, B=Bright spot

face_id = input('\n Enter user ID>>')
print("\n Initialiseng face capture. look at the camera and wait...")
count=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
# detectMultiscale(image, scalefactor, min_neighbours, flags, (minimum size, maximum size))
# it detects on basis of data fed by cascade file(here it is haarcascade feeds frontal face data)
# image = still image/ video feed - where you have to detect face
# scale_factor = it scales down the main webcam feed (here we have written 1.3 which means scale down by 30%)
# flags = this is for older versions of OPENCV
# min neighbours = default:5. it is the number of features the functions confirms whether it is a face or not. (here it will first match the 5 face
#                  features and when they match it will show result)
# min size, max size = Faces outside this range will not be detected(Ex. (200,250) this means faces only more than 200 pixels and less than 250
#                      pixels will be detected.

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count+=1
        cv2.imwrite("dataset/User."+str(face_id)+'.' +str(count)+".jpg", gray[y:y+h, x: x+w])
#imwrite>> it can save files and create filename but cannot create a file/folder
        cv2.imshow('image', img)
    k=cv2.waitKey(1)
    if k==27:
        break

print("\n Exiting program")
cam.release()
cv2.destroyAllWindows()

# THINGS TO REMEMBER
# create a folder 'dataset', in the project folder create a folder 'trainer', copy "haarcascade_frontalface_default.xml" in project folder

# this is aggregator code
import cv2
import numpy as np

from PIL import image
# PIL = Pythin Image Library( application:- it provides editing capabilities for all images)
# image is a class of PIL. it is responsible for rotation, saving, changing size, obtaining image path, renaming image path, etc.
import os

path='dataset'
recogniser = cv2.face.LBPHFaceRecognizer_create()
# it creates a face recogniser (inbuilt)
# Note: MTCNNFacenet is a better recogniser in same amount of training images(dataset)
#       we can also use cv2.face.MNTCNNFacenet_create() (main problem is that it is paid)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# this file is directly in project folder

def getImagesAndLables(path):
    imagePath = [os.path.join(parh, f) for f in os.listdir(path)]
# os.path.join is an example where we are interfering with os/drive.
# os.path.join (path, f)>> it concatnates path and f. Eg. os.path.join(C:, User) output will be C:\Users
# os.listdir(path)>> Lists the files within the path. Eg. os.pathdir(C:) will result Program Files, Users, Windows, etc

# os.path.getatime (path)>> it gives last time when path was accessed
# os.path.getsize (path)>> it gives size of path in bytes
    faceSamples = []
    ids =[]
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
# Image.open(imagePath).convert('L') >> image.convert(mode, matrix, dither, palette, colours)
#                                       mode: normal image
#                                       matrix: P='RGB to RGB', L='RGB to grayscale', 1='normal image to normal grained image(FOR DISTINCTION OF FOREGROUND AND BACKGROUND)'
#                                       dither:
#                                       palette: WEB/ADAPTIVE(it can only be used when you have selected your matrix as P)
#                                       colours: it defines how many colours to be used only when matrix is P. by default 256 colours are there
        img_numpy = np.array(PIL_img, 'uint8')
# np.array(image, type)>> it converts the commplete image into numerical array data. np-> numpy=numerical python
# uint8>> it is the unsighed interger in 8 bits(comes with numpy, not in-built in python)

        id = int(os.path.split(imgPath)[-1].split(".")[1])
# os.path.split>> it splita path into two parts, 1. Head path[denoted as 1] and 2. Tail path[denoted as -1]. (Eg. Users/AJ/file.txt>> User/AJ=Head file, file.txt=Tail file)
#                 Eg. imagepath = C:/Users/AJ/Users.01.01.jpg>> C:/Users/AJ=Head and Users.01.01.jpg=Tail
#                 (if we again split head then it takes letters but if we again split tail then it only takes numbers
#                 here when we split again, we are splitting from '.'. so head=User and tail=01.01, again split and we remain with 01 

        faces - detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y: y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("\n Training faces. it will take few seconds....")
faces, ids= getImageAndLables(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print("\n {0} faces trained. Exiting program.".format(len(np.unique(ids))))

# cv2.putText(arg)
# takes arguments>> image, text, organisation, font, fontscale, color, thickness, lineType
# it is used for writting something on the video feed
#image>> which image/video feed you want to write onto
#text>> what you want to write
#organisation>> botton_left_corner of text string
#font>> font style
#font scale>> size of font
#color>> color of text
#thickness>> text thickness
#line type>> width of upper and inner lines can be set

#FONT>> FONT_HERSHEY_COMPLEX
#       FONT_HERSHEY_SIMPLEX
#       FONT_HERSHEY_PLAIN
#       FONT_HERSHEY_PLAIN
#       FONT_HERSHEY_DUPLEX
#       FONT_HERSHEY_TRIPLEX
#       FONT_HERSHEY_COMPLEX_SMALL
#       FONT_HERSHEY_SCRIPT_SIMPLEX
#       FONT_HERSHEY_SCRIPT_COMPLEX>> IF YOU ADD '_FONT_ITALIC' OR '_FONT_BOLD' AFTER THIS IT WILL HAPPEN

#CODE=3

import cv2
import numpy as np
import os
recogniser = cv2.face.LBPHFaceRecogniser_create()
recogniser.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id=0
names= ['None', 'John', 'Akshat', 'Pratik', 'Adit', 'Santosh']
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbours=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,))
        id, confidence = recogniser.predict(gray[y:y+h, x:x+w])

        if(confidence<100):
            id = names[id]
            confidence = "{0}%".format(round(100-confidence))

        else:
            id = "unknown"
            confidence = "{0}%".format(round(100-confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 0, 0), 1)
    cv2.imshow('Detection', img)
    k = cv2.waitKey(1)
    if k== 27:
        break
print("\n Recognition in process")
cam.release()
cv2.destroyAllWindows()

# our face recog folder should have: 1. dataset folder, 2. trainer folder, 3. .py file and 4. haarcascade_frontalface_default.xml
# in that .py file type code 1, run it, it will ask for id, enter 1, then a webcam window will open up
# and take 30 snaps of you and then get closed automatically

# then remove code 1 from that file and type code 2 in same file and run it, it will not open a window
# a message will show how many faces are trained.

# then remove code 2 from the file and write code 3 in same file and run it, it will recognise your face
# and tell your name and error value. NOTE: In code 3, line 11: change names accordingly(id starts from 0)

# NOTE: after running code 1 and after 30 snaps are taken and window closes automatically, you can go
# to your dataset folder, it will contain 30 pics of you cropped from your face in grayscale font

# NOTE: After running code 2 and after it shows number of faces trained, you can go and check trainer folder
# you will find a file named trainer.yml

# NOTE: if the images are rotated or webcam is flipped you can use the code cv2.flip(img, -1)

