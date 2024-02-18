import pathlib
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
from PIL import Image
import torch
import numpy as np
import os
from taipy import Gui
import taipy as tp
from taipy import Config, Core
import taipy.gui.builder as tgb
import serial 
import time 

users = []

arduino = serial.Serial(port="COM7", baudrate=9600, timeout=.05)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the MTCNN face detection model
mtcnn = MTCNN(keep_all=True, device=device)



def compare(imageone, imagetwo):
    img1 = Image.open(imageone)
    img2 = Image.open(imagetwo)
    faces1 = mtcnn(img1)
    faces2 = mtcnn(img2)

    # If no faces are detected, return False
    if faces1 is None or faces2 is None:
        return False

    # Extract face embeddings
    emb1 = facenet_model(faces1.to(device)).detach().cpu().numpy()
    emb2 = facenet_model(faces2.to(device)).detach().cpu().numpy()

    # Calculate the cosine similarity between the embeddings
    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))

    # Define a similarity threshold (adjust as needed)
    threshold = 0.6

    print(similarity)

    # Return True if the similarity is above the threshold, indicating the same person
    return similarity > threshold



  




def docamera():
  workede = 2
  iterator = 0
  goes = 0

  cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

  clf = cv2.CascadeClassifier(str(cascade_path))

  camera = cv2.VideoCapture(0)
  times = 0
  while True and workede >= 2:
    times+=1
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    for (x, y, width, height) in faces:
      times+=1
      face_image = frame[y:y+height, x:x+width]
      face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
      face_gray_resized = cv2.resize(face_gray, (gray.shape[1], gray.shape[0]))
      #cv2.imwrite(f"facecamyy.jpg", face_gray_resized)
      cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
      camerar = cv2.VideoCapture(0)
      return_value, image = camera.read()
      string_name = 'photon.png'
      cv2.imwrite(string_name, face_image)
      if times%2 == 0:
        goes += 1
        try:
          if compare(string_name, 'photo.png'):
            iterator += 1
            print("YES")
          else:
            print("NO")
        except ValueError:
          print("NAHHH")
      if goes >= 4:
        if iterator >= 4:
          print("YAAA")
          workede = 1
        else:
          print("NOOOO")
          workede = 0
        break
      
      

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) == ord("q"):
      break

  camera.release()
  cv2.destroyAllWindows()

  if workede == True:
    return True
  return False

print("ok now lets start")
  


def taipy_app_found(img, name, desc):
  description = desc
  text_variable = name
  image_path = img  # Example placeholder image
    
    # Dynamically insert the variable content into the HTML
  html_content = f"<b>{text_variable}</b><br /><span>{description}</span><br /><img src='{image_path}' alt='Image' />"
    
  Gui(page=html_content).run(dark_mode=True)

#taipy_app_found("photo.png", "Sachin", "I am your best friend")


def not_found(img, name, desc):
  description = desc
  text_variable = name
  image_path = img  # Example placeholder image
    
    # Dynamically insert the variable content into the HTML
  html_content = f"<input>{text_variable}</input><br /><input>{description}</input><br /><img src='{image_path}' alt='Image' />"
    
  Gui(page=html_content).run(dark_mode=True)

# Example usage
#not_found("photo.png", "Unknown", "Type In")

def write_read(x):
       arduino.write(x.encode())
       return

name = "Unknown"
if docamera():
 print("YENNLWSNLWS")
 name = "Sachin"
 relation = " - Brother"
 name = name+relation
 write_read(name)
 taipy_app_found("photon.png", "Sachin", "I am your best friend")
else:
    not_found("photon.png", "Unknown", "Type In")
    name = "Unknown"
    


