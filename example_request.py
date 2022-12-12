
import time
import io
import json
import pprint
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
def ROI(img,x,y,x_w,y_h):
    if x>x_w:
        x,x_w=x_w,x
    if y>y_h:
        y,y_h=y_h,y
    return  img[y:y_h+1,x:x_w+1]


lasttime=time.time()
DETECTION_URL = "http://localhost:4003/v1/object-detection/custom"
  


IMAGE = "D://docker-test//yolov5-master//utils//docker//docker-test//Imagesave_3.png"
DEP_IMAGE="D://docker-test//yolov5-master//utils//docker//docker-test//Imagedepth_3.png"
# Read image
with open(IMAGE, "rb") as f: 
    image_data = f.read()
with open(DEP_IMAGE, "rb") as f:
    depth_data = f.read()
# payload={"image": image_data}
# payload=json.dumps(payload)
img=Image.open(DEP_IMAGE)
response = requests.post(DETECTION_URL,files={"imagefile": image_data,"depthfile":depth_data}).json()

#response = requests.post(DETECTION_URL).json()
rgb_image=np.asanyarray(Image.open(IMAGE))
depth_image = np.asanyarray(img)
print(depth_image)
pprint.pprint(response)
index=1
print(time.time()-lasttime)


for item in response["predictions"]:
    xmax,ymax,xmin,ymin=item["xyxy"]
    #int(item["xmax"]),int(item["xmin"]),int(item["ymax"]),int(item["ymin"])
    print(xmax,xmin,ymax,ymin)
    img1=ROI(rgb_image,xmax,ymax,xmin,ymin)
    img2=ROI(depth_image,xmax,ymax,xmin,ymin)
    plt.subplot(4,6,index), plt.title(item["class_name"])
    plt.imshow(img1),plt.axis('off')
    plt.subplot(4,6,index+1), plt.title(item["class_name"])
    plt.imshow(img2),plt.axis('off')
    index+=2
    if index >24 :break
plt.show()




