# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:52:52 2021

@author: Admin
"""
from mtcnn_cv2 import MTCNN
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
import models.CDCNs as CDCNs
from utils.utils import read_cfg
from utils.eval import predict

PATH = r"experiments\output\CDCNpp_nuaap.pth"
test_image = r"data\test\00_20230127004339.jpg"
cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
transform = transforms.Compose([
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])

model = CDCNs.CDCNpp()
model_main = torch.load(PATH)
#then using load_state_dict the parameters are loaded, strict is added 
#to prevent the error because of deleting part of the model
print(model_main['epoch'])
model.load_state_dict(model_main['state_dict'], strict=True)
detector = MTCNN()

#read image
img = cv2.imread(test_image)
# img1 = Image.open(test_image)
#change colour space to rgb for the mtcnn
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#finally the resulting faces properties are assigned to a variable 
faces = detector.detect_faces(image_rgb)


#if there is a face present to the following:
for face in faces:
    #extract keypoint and bounding box information from the result variable
    keypoints = face['keypoints']
    bounding_box = face['box']
    
    #assign bounding box infromation to proper variables for ease of use                    
    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = bounding_box[0]+bounding_box[2]
    y2 = bounding_box[1] + bounding_box[3]  
    
    edge=max(y2-y1,x2-x1)
    y1=int((y1+y2-edge)/2)
    y2=int((y1+y2+edge)/2)
    x1=int((x1+x2-edge)/2)
    x2=int((x1+x2+edge)/2)  

    cv2.rectangle(img, (x1, y1), (x2, y2),(50,255,0),2)
    # plt.imshow(img)
    # cv2.waitKey(0)


    #crop the image to include only the face section
    image_cr = image_rgb[y1:y2,x1:x2]

    # image_cr=cv2.resize(image_cr,(256,256))
    # img = img.transpose(2,0,1)
    # image_cr = image_cr.transpose(2,1,0)
    image=Image.fromarray(image_cr)
    image = transform(image).unsqueeze(0)
    # Now apply the transformation, expand the batch dimension, and send the image to the GPU
    # image = torch.from_numpy(img).float().unsqueeze(0)#.cuda(0)
    #after the image is in the proper form, it is loaded to the CDCNN model for output
    outputs = model(image)

    #torch.no_grad() is used to prevent errors from clashes 
    #and only output's [0] tuple part used out of the 6 since that 
    #part is the depth map which is the wanted output.
    with torch.no_grad():
        #only the x and y axis are used since depth is only 1 
        #by taking the mean of the depth map, the result of genuineness is found
        depth_map=outputs[0]
        depth_map=np.where(depth_map<1,depth_map,1)
        # score = torch.mean(depth_map, axis=(1,2))
        score = np.mean(depth_map, axis=(1,2))
        # preds, score = predict(depth_map)
        #for finding the pixel number
        #score = torch.sum(outputs[0] > 0.05) 
        print(score)

        #depth map part, changes output to image shape and after transpose, suitable
        #to be shown with cv2
        # depth_bin_img = outputs[0].cpu().numpy()
        # new = depth_bin_img.transpose(2,1,0)
        # cv2.imshow("depth map", new)
        # cv2.waitKey(0)
            
    #if the resulting score is bigger than 0.6 it is genuine otherwise it is fake     
    if score >= 0.5: # or 150 if going from pixel number
        result = "1_live"
        print("live")
    else:
        result = "0_spoof"
        print("spoof")
        
    #to print genuine or not at the bottom of the window
    cv2.putText(img, str(result), (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(img, str(round(float(score),4)), (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

#this parts are for showing the rseults, the image and the depth map
cv2.imshow("mtcnn face",img)
cv2.waitKey(0)


cv2.destroyAllWindows()