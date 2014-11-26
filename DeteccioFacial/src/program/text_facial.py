'''
Created on 26/11/2014

@author: lpheras
'''

import cv2

if __name__ == '__main__':
    pass



in_text_file1 = "./files2.txt"
in_text_file2 = "./out.txt"
face_cascade = cv2.CascadeClassifier('D:\\opencv\\sources\\data\\\haarcascades\\haarcascade_frontalface_alt2.xml')

f1=open(in_text_file1)
data_image=f1.readlines()

f2=open(in_text_file2)
data_bb=f2.readlines()

i = 0;
for line_image in data_image:
    line_bb = data_bb[i];
    line_bb = line_bb.replace(':\n','');
    line_image = line_image.replace('\n','');
    image = cv2.imread(line_image)
    image = cv2.pyrDown(image);
    
    bb_parts = line_bb.split(":")
    tx = image.shape[0]; bx = -1;
    ly = image.shape[1]; ry = -1;
    for part in bb_parts:
        par = part.split(";")
        for pa in par:
            p = pa.split(",") 
            if tx > int(p[0]): tx = int(p[0])
            if bx < int(p[0]): bx = int(p[0])
            if ly > int(p[1]): ly = int(p[1])
            if ry < int(p[1]): ry = int(p[1])
    
    #faces = face_cascade.detectMultiScale(image, 1.3, 10, 20)
    faces = face_cascade.detectMultiScale(image, 1.3,2,0,(50,50),(200,200))

    if len(faces)>0:
        area = 0;
        x1 = 0; y1 = 0; x2 = 0; y2 = 0;
        for x,y,w,h in faces:
            if w*h > area:
                x1 = x; x2 = x+h; y1 = y; y2 = y+w;
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        if tx > x1: tx = x1;
        if bx < x2: bx = x2;
        if ly > y1: ly = y1;
        if ry < y2: ry = y2;
        
    cv2.rectangle(image,(tx,ly),(bx,ry),(255,0,0),2)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    i+=1    
        
        
        
        
        
        
        
    