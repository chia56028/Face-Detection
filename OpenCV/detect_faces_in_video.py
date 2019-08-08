import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade_profile = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')


_CAMERA_WIDTH = 640
_CAMERA_HEIGH = 480

# Specify which camera to use
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMERA_HEIGH)


while(True):
    # Capture frame-by-frame
    # ret will receive True/False
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Front face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    n = 0 
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        img_item = "image_"+str(n)+".png"
        n += 1
        cv2.imwrite(img_item, roi_gray)

        # draw a rectangle on frame
        color = (0,0,255) # BGR 0-255
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

    # Right face
    faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    n = 0 
    for (x,y,w,h) in faces:
    	print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = gray[y:y+h, x:x+w]
    	img_item = "image_right_"+str(n)+".png"
    	n += 1
    	cv2.imwrite(img_item, roi_gray)

    	# draw a rectangle on frame
    	color = (255,0,0) # BGR 0-255
    	stroke = 2
    	cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)


    # Left face
    flipped = cv2.flip(gray, 1)
    faces = face_cascade_profile.detectMultiScale(flipped, 1.5, 5)

    n = 0 
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, _CAMERA_WIDTH-x-w:_CAMERA_WIDTH-x]
        roi_color = gray[y:y+h, _CAMERA_WIDTH-x-w:_CAMERA_WIDTH-x]
        img_item = "image_left_"+str(n)+".png"
        n += 1
        cv2.imwrite(img_item, roi_gray)

        # draw a rectangle on frame
        color = (0,255,0) # BGR 0-255
        stroke = 2
        cv2.rectangle(frame, (_CAMERA_WIDTH-x-w,y), (_CAMERA_WIDTH-x,y+h), color, stroke)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()