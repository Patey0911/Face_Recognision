import cv2

eyes_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
_,frame1=cap.read()

while(True):
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame1[y:y+h, x:x+w]
        eyes=eyes_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)


    cv2.imshow('window', frame1)
    _,frame1=cap.read()
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cv2.destroyAllWindows()