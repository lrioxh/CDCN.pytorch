from mtcnn_cv2 import MTCNN
import cv2, os
# import pandas as pd

dir=r"./data/custom"
data_dir=r"./data/nuaa/images"

detector = MTCNN()
# train_list=pd.read_csv('./data/nuaa/train.csv')
train_list=open('./data/nuaa/train.csv','a')

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


for file in os.listdir(dir):
    if file.endswith('.txt'):continue
    label=int(file.split('_')[0])   #1=genuine/live

    path=os.path.join(dir,file)
    img_origin = cv2.imread(path)
    #data generated by rotate & flip
    imgs=[img_origin,rotate(img_origin,10),rotate(img_origin,-10),cv2.flip(img_origin,1)]  
    item=0
    for img in imgs:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        
        for face in faces:

            keypoints = face['keypoints']
            bounding_box = face['box']
            
                    
            x1 = bounding_box[0]
            y1 = bounding_box[1]
            x2 = bounding_box[0]+bounding_box[2]
            y2 = bounding_box[1] + bounding_box[3]  

            image_cr = img[y1:y2,x1:x2]
            edge=max(y2-y1,x2-x1)

            image_resize1=cv2.resize(image_cr,(edge,edge))
            image_resize1=cv2.resize(image_resize1,(256,256))

            y1=max(int((y1+y2-edge)/2),0)
            y2=int((y1+y2+edge)/2)
            x1=max(int((x1+x2-edge)/2),0)
            x2=int((x1+x2+edge)/2)  
            image_resize2=img[y1:y2,x1:x2]
            image_resize2=cv2.resize(image_resize2,(256,256))

            # cv2.imshow('',image_resize1)
            # cv2.waitKey(0)
            # cv2.imshow('',image_resize2)
            # cv2.waitKey(0)
            name1=f"{file.split('.')[0]}_{item}_.jpg"
            name2=f"{file.split('.')[0]}_{item+1}_.jpg"
            cv2.imwrite(os.path.join(data_dir,name1),image_resize1)
            cv2.imwrite(os.path.join(data_dir,name2),image_resize2)
            item+=2
            train_list.write(f"images/{name1},{label}\n")
            train_list.write(f"images/{name2},{label}\n")
        # break
    print(f"{path} added")
    # 4*2 samples per face

train_list.close()