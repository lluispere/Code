'''
Created on 26/11/2014

@author: lpheras
'''

import cv2
import numpy as np

if __name__ == '__main__':
    pass



in_text_file1 = "./scanner.txt"
in_text_file2 = "./scannerout.txt"
face_cascade = cv2.CascadeClassifier('D:\\opencv\\sources\\data\\\haarcascades\\haarcascade_frontalface_alt2.xml')

f1=open(in_text_file1)
data_image=f1.readlines()

f2=open(in_text_file2)
data_bb=f2.readlines()

i = 0;
for line_image in data_image:
    line_bb = data_bb[i];
    line_image = line_image.replace('\n','');
    image = cv2.imread(line_image)
    image = cv2.pyrDown(image);
    if not line_bb == "\n": 
        line_bb = line_bb.replace(':\n','');
                
        bb_parts = line_bb.split(":")
        tx = image.shape[0]; bx = -1;
        ly = image.shape[1]; ry = -1;
        for part in bb_parts:
            par = part.split(";")
            #for pa in par:
            pa1 = par[0];
            pa2 = par[1];
            
            p = pa1.split(",")
            x1 = int(p[0])
            y1 = int(p[1])
            if tx > int(p[0]): tx = int(p[0])
            if ly > int(p[1]): ly = int(p[1])
            p = pa2.split(",")
            x2 = int(p[0])
            y2 = int(p[1])
            if bx < int(p[0]): bx = int(p[0])
            if ry < int(p[1]): ry = int(p[1])
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),2)
        
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
        print bx
        if not bx == -1:
            mask = np.zeros(image.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            rect = (tx//2,ly//2,bx+((image.shape[1]-bx)//2),ry+((image.shape[0]-ry)//2)) 
            #rect2 = (232, 37, 1000, 712)
            
            cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),2)
            #cv2.rectangle(image,(rect2[0],rect2[1]),(rect2[2],rect2[3]),(0,255,255),2)
            #print rect2
            #print (rect2[0],rect2[1]),(rect2[2],rect2[3])
            cv2.imshow("",image)
            cv2.waitKey(0)
            print rect  
            cv2.imwrite("D:\\ICAR\\outin\\"+str(i)+".jpg",image)
                 
            cv2.grabCut(image,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
        
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            output = cv2.bitwise_and(image,image,mask=mask2)
            cv2.imshow("",output)
            cv2.waitKey(0)
            #cv2.rectangle(image,(tx,ly),(bx,ry),(255,0,0),2)
            #cv2.imshow('img',image)
            #cv2.waitKey(0)'''
    else:
        cv2.imwrite("D:\\ICAR\\outin\\"+str(i)+".jpg",image) 
    i+=1    
        
        
        
        
        
        
        
    