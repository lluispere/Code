'''
Created on 19/11/2014

@author: lpheras
'''
import cv2
import os
import numpy as np
import glob
import csv
from numpy import argmin

if __name__ == '__main__':
    pass
icar = '..\\..\\..\\..\\'
folder_train = icar+'train\\*'
folder_test = icar+'test\\*'
csvPathTrain = icar+'GroundTruth\\train.csv'
csvPathTest = icar+'GroundTruth\\test.csv'
outPath = icar+'Results\\ReconeixementFacial\\'

sub_folders_train = glob.glob(folder_train)
sub_folders_test = glob.glob(folder_test)
face_cascade = cv2.CascadeClassifier('D:\\opencv\\sources\\data\\\haarcascades\\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# carregar GT
with open(csvPathTrain, "rb") as csvFile:
    xls = csv.reader(csvFile, delimiter=";")
    with open(csvPathTest, "rb") as csvFile2:
        xlstest = csv.reader(csvFile2, delimiter=";")
        
        # per cada carpeta de train
        for f in sub_folders_train:
            fname = f.split('\\')[5]
            models = glob.glob(f+'\\*.jpg')
            #carreguem el model
            if len(models) > 1:
                #imatge
                #model = cv2.pyrDown(cv2.imread(models[0]))
                model = cv2.imread(models[0])
                model_name = models[0].split('\\')[6]
                #punts GT
                csvFile.seek(0)
                modelp = [np.asarray(row[2:10],np.float32) for row in xls if row[0]==model_name]
                modelp = np.asarray(modelp).reshape(-1)
                mp1 = (0,0)
                mp2 = (model.shape[1],0)
                mp3 = (model.shape[1],model.shape[0])
                mp4 = (0,model.shape[0])
                modelp = [mp1,mp2,mp3,mp4]
            #per cada imatge al test
            for f2 in sub_folders_test:
                f2name = f2.split('\\')[5]
                #nomes en el cas de que es tracti de la mateix classe que el train
                if fname == f2name:
                    images = glob.glob(f2+'\\*.jpg')
                    hom = []
                    points = []
                    #per cada imatge
                    for img_file in images:
                        img_name = img_file.split('\\')[6]
                        csvFile2.seek(0)
                        imgp = [np.asarray(row[2:10],np.float32) for row in xlstest if row[0]==img_name]
                        imgp = np.asarray(imgp).reshape(-1)
                        imgp[(imgp<0).nonzero()] = 0
                        ip1 = (imgp[0],imgp[1])
                        ip2 = (imgp[2],imgp[3])
                        ip3 = (imgp[4],imgp[5])
                        ip4 = (imgp[6],imgp[7])
                        imgp = [ip1,ip2,ip3,ip4]
                        #img = cv2.pyrDown(cv2.imread(img_file))
                        img = cv2.imread(img_file)
                        faces = face_cascade.detectMultiScale(img, 1.3, 10)
                        if len(faces)>0:
                        
                            for x,y,w,h in faces:
                                cv2.rectangle(img,(x,y),(x+h,y+w),(255,0,0),2)
                                #cv2.imshow('img',img)
                                #cv2.waitKey(0)
                            
                            # calcul de la homografia
                            print modelp
                            print imgp
                            hom = cv2.getPerspectiveTransform(np.float32(imgp),np.float32(modelp),)
                            p = np.array([[x,y],[x+h,y],[x,y+w],[x+h,y+w]],dtype = 'float32')
                            p = np.array([p])
                            
                            points = cv2.perspectiveTransform(p,hom)
                            print points
                            
                            points = points.reshape(-2)
    
                            cv2.rectangle(model,(points[0],points[1]),(points[6],points[7]),(255,0,0),2)
                            #cv2.imshow('img',model)
                            #cv2.waitKey(0)


                    cv2.imwrite(outPath+model_name,model)
                    break

    
    
    
    
    
    