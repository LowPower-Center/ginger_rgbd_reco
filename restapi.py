# RestAPI by LowPower-Center

import argparse
import io
import numpy as np
import torch
from flask import Flask, request
from PIL import Image
from utils import *
import cv2
import math
import pprint
import time
import os
#######################################################################################
#---------------------------------------utils-----------------------------------------#
#######################################################################################

def eulerAnglesToRotationMatrix(angles1) :
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*3.141592653589793/180.0
    theta[1] = angles1[1]*3.141592653589793/180.0
    theta[2] = angles1[2]*3.141592653589793/180.0
    print(theta)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ]) 
    return np.dot(R_z, np.dot( R_y, R_x ))



def list_to_JSON(results):
    JSONlist=[]
    if mat_flag:
        for xyxy,conf,cls_name,pos,base_pos,coordinate in results:
            item={
        "xyxy": xyxy,
        "class_name": cls_name,
        "confidence": conf,
        "position":pos,
        "relative_pos":base_pos,
        "coordinate":coordinate}
            JSONlist.append(item)
    else:
        for xyxy,conf,cls_name,pos in results:
            item={
        "xyxy": xyxy,
        "class_name": cls_name,
        "confidence": conf,
        "position":pos}
            JSONlist.append(item)
    
    JSONdict={"predictions":JSONlist,"mat_flag":mat_flag}
    return JSONdict


def ROI(img,x,y,x_w,y_h):
    if x>x_w:
        x,x_w=x_w,x
    if y>y_h:
        y,y_h=y_h,y
    return  img[y:y_h+1,x:x_w+1]


def calculate_depth(pred,depth_img):
    results=[]
    for item in eval(pred):
        # get xyxy
        xmax,xmin,ymax,ymin=int(item["xmax"]),int(item["xmin"]),int(item["ymax"]),int(item["ymin"])
        x_center,y_center=(xmax+xmin)/2,(ymax+ymin)/2
        # get depth ROI
        depth=ROI(depth_img,xmax,ymax,xmin,ymin)
        depth=depth[depth!=0]
        # depth = median depth of ROI
        depth_pos = np.median(depth,axis=None)
        #print(depth_pos)
        #print(x_center,y_center)
        # get pixel coordinate
        pixel_pos=np.array([x_center,y_center,1])*depth_pos.item()
        # calculate camera coordinate
        pos = list(np.linalg.solve(camera_intrinsics,pixel_pos))
        # prepare for json format
        xyxy=[xmax,ymax,xmin,ymin]
        conf=item["confidence"]
        cls_name=item["name"]
        results.append([xyxy,conf,cls_name,pos])
    return results


def get_para(mat_str):
    global mat_flag
    mat_flag=True
    x=mat_str.split("_")
    X=eval(x[1].split("=")[-1])
    Y=eval(x[2].split("=")[-1])
    Z=eval(x[3].split("=")[-1])
    T=np.array((X,Y,Z))
    P=eval(x[5].split("=")[-1])/180*np.pi
    Y=eval(x[6].split("=")[-1])/180*np.pi
    R=eval(x[7].split("=")[-1])/180*np.pi
    Rotate=eulerAnglesToRotationMatrix((R,P,Y))

    start_x=eval(x[11].split("=")[-1])
    start_y=eval(x[12].split("=")[-1])
    faceto_x=eval(x[13].split("=")[-1])
    faceto_y=eval(x[14].split("=")[-1])
    faceto_z=eval(x[15].split("=")[-1])
    start_z=faceto_z
    pos_start=(start_x,start_y,start_z)
    pos_faceto=(faceto_x,faceto_y,faceto_z)
    print(Rotate,T,pos_start,pos_faceto)

    return Rotate,T,pos_start,pos_faceto



def calculate_coordinate(para):
    points=para
    for index,item in enumerate(points):
        #print(item)
        pos=np.array(item[3]).reshape(-1,1)
        #print(pos)
        pos=np.matmul(Rotation_p2c,pos)
        pos=np.matmul(Rotation_c2b,pos)+Translation_c2b.reshape(-1,1)
        relative_pos=[float(x) for x in list(pos)]
        points[index].append(relative_pos)
        coordinate=list(np.matmul(Rotation_b2w,pos)+Translation_b2w.reshape(-1,1))
        coordinate=[float(x) for x in coordinate]
        points[index].append(coordinate)
    return points

def find_hitting_point(results,target):
    assert mat_flag
    cb=get_target_ball(results,"b0")
    tb=get_target_ball(results,target)
    pos=get_hitting_position(cb,tb)
    return pos,cb["coordinate"]

def get_hitting_position(cb,tb):
    global pos_faceto
    cb_x,cb_y,cb_z=cb["coordinate"]
    tb_x,tb_y,tb_z=tb["coordinate"]
    assert 0.5 < tb_z/cb_z <2 
    scale=1000 # means cm  ,while 1000 means mm
    vec=np.array([cb_x-tb_x,cb_y-tb_y])
    dis=np.sqrt((cb_x-tb_x)**2+(cb_y-tb_y)**2)
    pos=vec/dis*scale+np.array([cb_x,cb_y])
    pos_x,pos_y=list(pos)
    return pos_x,pos_y,930
    
    

def get_target_ball(results,target="any"):
    if target=="any":
        tg_list=[item for item in results if item["class_name"] != "b0"]
    else:
        tg_list=[item for item in results if item["class_name"] == target]
    conf_list=[item["confidence"] for item in tg_list]  
    return tg_list[np.argmax(conf_list)]
   
#######################################################################################
#---------------------------------------route-----------------------------------------#
#######################################################################################


# here is config variety
app = Flask(__name__)
models = {}
TEST_URL = "/test"
REQUEST_URL = "/v1/find_hitting_point/<target>"
DETECTION_URL = "/v1/object-detection/<model>"
CALCULATION_URL = "/v1/point-calculation/<mat>"
camera_intrinsics=np.matrix('378.307,0,320.381;0,377.500,243.394;0,0,1')  


#Rotation_b2w=np.matrix('0,-1,0;1,0,0;0,0,1')
Rotation_b2w=np.matrix('1,0,0;0,-1,0;0,0,1')
Rotation_p2c=np.matrix('0,0,1;-1,0,0;0,-1,0')   # x right y down
#Rotation_p2c=np.matrix('0,0,1;0,1,0;-1,0,0')   #x down
Translation_b2w=np.array([114.122, 416.829, 0])*10
T1=np.array([114.122, 416.829,  93. ])*10
pos_faceto=(0,0,930)
pos_start=(0,0,930)
mat_flag=True
global_results=""
i=1
# [[ 5.98942664e-17 -1.00000000e+00  1.27309193e-17]
#  [ 9.78147601e-01  6.12323400e-17  2.07911691e-01]
#  [-2.07911691e-01  0.00000000e+00  9.78147601e-01]]

Rotation_c2b=np.matrix('0,-1,0;0.9781476,0,0.2079;-0.2079,0,9.7814e-01')
#Rotation_c2b=np.matrix('9.99617537e-01,-2.74121931e-02,-3.65402845e-03;2.74120098e-02,9.99624215e-01,-1.00225443e-04;3.65540272e-03,2.28461538e-08,9.99993319e-01')
Translation_c2b=np.array([-1.039 , 12.942 ,155.969])*10

@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    # global Rotation
    # global Translation
    global global_results
    if request.method != "POST":
        return
    i=1
    cpath = os.getcwd()
    print("Path =", cpath)
    if request.files.get("imagefile"):
        #read RGB file
        im_file = request.files["imagefile"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        # a=int(time.time()//20000)
        # file_path="imagesave_{}.png".format(a)
        # im.save(file_path)
        #read Depth file
        dp_file = request.files["depthfile"]
        dp_bytes = dp_file.read()
        dp = Image.open(io.BytesIO(dp_bytes))
        depth_img=np.array(dp)
        #print(depth_img.max())
        # cv2.imwrite("dep_img_{}.png".format(a),dp)

        # dp.save("depth"+file_path)
        # predict RGB image
        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference

        # get prediction
        pred=results.pandas().xyxy[0].to_json(orient="records")

        # calculate depth 
        pos=calculate_depth(pred,depth_img)

        # calculate world coordinate if program get outer param
        if mat_flag:
            pos=calculate_coordinate(pos)
        
        # transform result format to JSON
        rlt=list_to_JSON(pos)
        global_results=rlt["predictions"]
        #print('success!!!!')
        #pprint.pprint(rlt)
        return rlt

# test varest module
@app.route(TEST_URL, methods=["POST"])
def test():   
    print("success")
    y={'predictions': [{'class_name': 'b0', 'confidence': '0.8404963017', 'position': '[-26.08678137068565, 212.08877350993373, 1052.0]', 'xyxy': [323, 299, 332, 307]}]}
    pprint.pprint(y)
    return y


# receive rotation matrix and translation vector from ginger
@app.route(CALCULATION_URL, methods=["POST"])
def calculate(mat):
    # global Rotation
    # global Translation
    global pos_start
    global pos_faceto
    if request.method != "POST":
        return
    print(mat)
    Rotation,Translation,pos_start,pos_faceto=get_para(mat)
    print("success!!!")
    return "Mission Complete"


# calculate hitting point for ginger and send to  ginger
@app.route(REQUEST_URL, methods=["POST"])
def request_for_hitting(target):   # target support  b1 b2 b3 b4 b5 or any
    global global_results
    print(global_results)
    try:   
        pos,cb_pos=find_hitting_point(global_results,target)
        print(pos,cb_pos)
        pos=list(pos)+list(cb_pos)
    except:
        print("I am here")
        return {"predictions": [{"valid":"1","pos":[14,387,93,74,307,93]}]}
    y={'predictions':[{'valid': "0",'pos':[x/10 for x in pos]}]}
    return y


if __name__ == "__main__":
    # default code ,I dare not to modify
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=4003, type=int, help="port number")
    parser.add_argument('--model', nargs='+', default=['custom'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()
    for m in opt.model:
        models[m] = torch.hub.load("./", m, path='./best.pt', source='local')
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
