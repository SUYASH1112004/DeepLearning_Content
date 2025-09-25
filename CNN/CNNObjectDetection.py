import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
import json


#Load pretrained CNN
model=MobileNetV2(weights="imagenet")

with open('imagenet_class_index.json','r') as f:
    class_index=json.load(f)



#Captures From Web Cam
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img_resized=cv2.resize(img,(224,224))
    x=np.expand_dims(img_resized,axis=0).astype(np.float32)
    x=preprocess_input(x)

    #Predictions
    preds=model.predict(x,verbose=0)
    pred_class=int(np.argmax(preds,axis=-1)[0])
    decoded=class_index[str(pred_class)]
    label=f"{decoded[1]} : {preds[0][pred_class]*100 :.1f} %"

    #Show on screen
    cv2.putText(frame,label,(16,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("CNN Classification ",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
