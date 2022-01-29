
import numpy as np
import cv2
import json
import os
import face_recognition
import sys

imglist = [] 
featurelist = []
files = []

path = sys.argv[-1]
pathlen = len(path)
#filecheck = "faceCluster_"
#ind = path.find(filecheck)
clusternum = path[pathlen-1]
noofclusters = int(clusternum)

for image in os.listdir(path):
    img = cv2.imread(path+"\\"+image)
    fcascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fcascade.detectMultiScale(gimg, 1.2, 4)
    for (x,y,w,h) in faces:
        x1=x+w
        y1=y+h
        img = cv2.rectangle(img,(x,y),(x1,y1),(0,255,0),2)
        imgdata={"iname":image, "bbox" :[float(x), float(y), float(w), float(h)]}
        imglist.append(imgdata)
        box = [(y,x1,y1,x)] # box = [(top,left,bottom,right)] as mentioned in the question
        #box = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
        bins = face_recognition.face_encodings(img,box)
        featurelist.append(np.array(bins))
        files.append(image)

feature = np.array(featurelist).reshape(len(imglist),128) 
#featurecoord = feature.T 
cluster = np.zeros(len(feature))
clusterarray = np.arange(noofclusters)

"""
Reference for K-means algorithm: https://www.geeksforgeeks.org/ml-k-means-algorithm/
"""
   
# sum of squared distance algorithm
def SSD(x, y):
    return np.sum((x - y)**2)

# initialization algorithm
def initcluster(imagedata, k):

    centroids = []
    centroids.append(imagedata[np.random.randint(imagedata.shape[0]), :]) # the first centroid is randomly selected
   
    for c_id in range(k - 1):   # now we can compute the values for the remaining (k-1) centroids
          
        dist = []
        for i in range(imagedata.shape[0]):
            point = imagedata[i, :]
            d = sys.maxsize     # the initial value of reference is set with a huge number so that the first centroid will always be minimum for future reference with other centroids
              
            for j in range(len(centroids)):
                tempd = SSD(point, centroids[j])    # finding the distance between the points
                d = min(d, tempd)   # finding the minimum distance w.r.t the reference
            dist.append(d)  # we update the list that stores distances of points from the nearest centroid
              
        dist = np.array(dist)
        next_centroid = imagedata[np.argmax(dist), :]    # we select the data point with the maximum distance as the next centroid value
        centroids.append(next_centroid)
        dist = []
    return centroids

centroids = initcluster(feature, noofclusters)
centroid = np.array(centroids)  # converting the list of centroids we obtained into an np array  
centroidarray = np.vstack(centroid) # np.vstack() coverts the above np array into vertical stacks with each row representing a centroid
centroidlist = list()

#n=0
#while True:
for n in range(0,200):   # hardcoded the total number of iterations to be 200
     for i in range(len(feature)):
         dis =  centroidarray - feature[i]         
         cluster[i] = np.argmin(np.sqrt(np.square(dis).sum(axis=1)),axis=0)
         
     for j in range(noofclusters):
         location = np.array(np.where(clusterarray[j]==cluster))        
         if not all(map(lambda x: all(x), location)):  #To access all values of W
           centroidarray[j,:] = np.average(feature[location,:], axis=1)
     centroidlist.append(centroidarray)
     #n = n+1
     if n>1 and centroidlist[n].all() == centroidlist[n-1].all(): #if the same centroid values are observed again, no need to iterate any further
         break
     
cluster = np.array(cluster+1).astype(int)
clusterarray = np.array(clusterarray+1).astype(str)
clusterlist = list(zip(cluster, files))
clusterlist.sort()
clusterfinal = np.array(clusterlist)

json_list = []
for j in range(noofclusters):
    index = np.array(np.where(clusterfinal[:,0]== clusterarray[j]))
    clist = (clusterfinal[index,1]).tolist()
    cldata={"cluster_no":clusterarray[j], "elements" :clist}
    json_list.append(cldata)

output_json = (os.getcwd()+".\\clusters.json")
with open(output_json, 'w') as f:
    json.dump(json_list, f)
