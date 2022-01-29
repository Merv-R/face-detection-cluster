
import cv2
import json
import os
import sys

json_list = [] 
path = sys.argv[-1]
#annopath = (".\Validationfolder\annotated\\")
#annopath = fpath+"\annotated\\"
imagepath = path+"\images"
#imagepath = ".\Validationfolder\images\\"
for image in os.listdir(imagepath):
    img = cv2.imread(imagepath+"\\"+image)
    fcascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fcascade.detectMultiScale(gimg, 1.2, 4)
    for (x,y,w,h) in faces:
        x1 = x+w
        y1 = y+h
        img = cv2.rectangle(img,(x,y),(x1,y1),(0,255,0),2)
        imgdata={"iname":image, "bbox" :[float(x), float(y), float(w), float(h)]}
        json_list.append(imgdata)
    #cv2.imwrite(annopath+image, img)    
output_json = (os.getcwd()+".\\results.json")
with open(output_json, 'w') as f:
    json.dump(json_list, f)
