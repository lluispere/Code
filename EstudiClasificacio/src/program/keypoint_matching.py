'''
Created on Oct 24, 2014

@author: lpheras
'''
import cv2
import numpy as np
import os
from datashape.coretypes import uint16
from cv2 import waitKey

orbdetector = cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)
flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)

outPath = "..\\..\\..\\..\\Results\\Classificacio_v5\\test500\\Confusions\\";

folders = os.listdir(outPath)
descriptors_train = [];
descriptors_test = [];

for folder in folders:
    # agafem les quatre imatges
    imglist = os.listdir(outPath+folder)
    if len(imglist)==4:
        img1p = outPath+folder+"\\"+imglist[0]
        img2p = outPath+folder+"\\"+imglist[1]
        img3p = outPath+folder+"\\"+imglist[2]
        img4p = outPath+folder+"\\"+imglist[3]
        
        # calculem els descriptors de train img1p
        #imgtrain1=cv2.pyrDown(cv2.imread(img1p))
        imgtrain1=cv2.imread(img1p)
        graytrain1 = cv2.cvtColor(imgtrain1, cv2.COLOR_BGR2GRAY)
        (keyptrain1,desctrain1)=orbdetector.detectAndCompute(graytrain1,None)
        
        # els afegim al matcher
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        matcher.add([desctrain1])
        
        # calculem els descriptors de trest img3p
        #imgtest1=cv2.pyrDown(cv2.imread(img3p))
        imgtest1=cv2.imread(img3p)
        graytest1 = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
        (keyptest1,desctest1)=orbdetector.detectAndCompute(graytest1,None)
        
        # calculem els matchings
        matches = matcher.knnMatch(desctest1, k = 2)
        matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75)]
        
        # en el cas que hi hagi mes d'un matching calculem el RANSAC
        if len(matches)>0:
            trainmatchesId = np.array([m.trainIdx for m in matches])
            querymatchesId = np.array([m.queryIdx for m in matches])
            # comencem RANSAC
            kptrainpt = []
            kpquerypt = []
            kptrain = []
            kpquery = []
            # agafem els descriptors d'interes
            for i in range(len(trainmatchesId)):
                kptrainpt.append(keyptrain1[trainmatchesId[i]].pt)
                kpquerypt.append(keyptest1[querymatchesId[i]].pt)
                kptrain.append(keyptrain1[trainmatchesId[i]])
                kpquery.append(keyptest1[querymatchesId[i]])
            
            # calculem la homografia
            hom,mask = cv2.findHomography(np.float32(kptrainpt),np.float32(kpquerypt),cv2.RANSAC,5.0)
            # comprobem que el RANSAC hagi trobat una homografia
            
            keypoints_matching_train = []
            keypoints_matching_test = []
            if not mask is None:
                for i in range(len(mask)):
                    # apuntem aquells descriptors que realment mostren una coherencia espaial
                    if mask[i]==1 : keypoints_matching_train.append(kptrain[i]),keypoints_matching_test.append(kpquery[i]) 
            
                #image = cv2.drawMatches(imgtrain1,keyptrain1,imgtest1,keyptest1,matches,imgtrain1)
                image = cv2.drawKeypoints(imgtrain1,keypoints_matching_train,imgtrain1)
                image2 = cv2.drawKeypoints(imgtest1,keypoints_matching_test,imgtest1)
                
                h1, w1, d1 = image.shape[:]
                h2, w2, d2 = image2.shape[:]
                vis = np.zeros((max(h1, h2), w1+w2, d1), np.uint8)
                vis[:h1, :w1, :d1] = image
                vis[:h2,w1:w1+w2,:d2] = image2
                
                # dibuixem les linees que connecten els keypoints
                for i in range(len(keypoints_matching_train)):
                    pt = (int(keypoints_matching_train[i].pt[0]),int(keypoints_matching_train[i].pt[1]))
                    newpt = keypoints_matching_test[i].pt
                    newpt = (int(newpt[0]+w1),int(newpt[1]))
                    cv2.line(vis,pt,newpt,(255,0,0))
                
                vis = cv2.pyrDown(vis)
                cv2.imwrite(outPath+folder+"matching.jpg",vis)
                
