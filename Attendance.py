#!/usr/bin/env python
# coding: utf-8

# In[34]:


import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime


# In[35]:


#automatic images loading 
path = 'Attendace_Images'
images = []
className = []
mylist = os.listdir(path)
mylist


# In[36]:


for cl in mylist:
    currentimage = cv2.imread(f'{path}/{cl}')
    images.append(currentimage)
    className.append(os.path.splitext(cl)[0])
    
print(className)


# In[37]:


#encoding 
def findencodings(images):
    encodlist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodlist.append(encode)
    return  encodlist

def markAttendance(name):
    with open ('Attendance.csv' , 'r+') as f :
        myDataList = f.readlines()
        nameList = []
        for line in  myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
encodeListKnown = findencodings(images)
print('Encoding Complete')


# In[ ]:





# In[38]:


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, img = vid.read()
    cv2.imshow('frame', img)
    imags = cv2.resize(img,(0,0),None , 0.25 , 0.25)
    
 ##   imags = resize(scale = 25 , img = img)

    imags=cv2.cvtColor(imags,cv2.COLOR_BGR2RGB)
    face_cur_frame = face_recognition.face_locations(imags)
    encode_current_frame = face_recognition.face_encodings(imags,face_cur_frame)
    print(face_cur_frame)
   
    for encodeface , locface in zip(encode_current_frame , face_cur_frame):
        matches = face_recognition.compare_faces(encodeListKnown ,encodeface)
        facedis = face_recognition.face_distance(encodeListKnown ,encodeface)
        matchIndex = np.argmin(facedis)
        
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print (name)
            y1,y2,x1,x2 = locface
            y1,y2,x1,x2 = y1*4,y2*4,x1*4,x2*4
            cv2.rectangle(img,(x2,y1),(y2 , x1 ), (0,255,0),2)
          ##  cv2.rectangle(img,(x2,y1-65 ),(y2 , x1), (0,255,0),cv2.FILLED)
       
            cv2.putText(img , name , (x1+6 , y2-6) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,255,255),2)
        
            markAttendance(name)  

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# In[ ]:


cv2.imshow('Webcam' , img)
cv2.waitKey(0)
  


# In[ ]:





# # 
