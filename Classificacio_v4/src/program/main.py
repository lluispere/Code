'''
Created on 27/10/2014

@author: lluispere
'''

import cv2
import glob
import numpy as np
from Docmodel import *
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import os

''' 
'' Aquesta funcio transforma els keypoints en una variable addient per a desar a disk 
'''
def convertirKeypointsToVars(keyp):
    tmppt = []
    tmpsize = []
    tmpangle = []
    tmpresponse = []
    tmpoctave = []
    tmpclassid = []
    
    for k in keyp:
        tmppt2 = []
        tmpsize2 = []
        tmpangle2 = []
        tmpresponse2 = []
        tmpoctave2 = []
        tmpclassid2 = []
        for k2 in k:
            tmppt2.append(k2.pt)
            tmpsize2.append(k2.size)
            tmpangle2.append(k2.angle)
            tmpresponse2.append(k2.response)
            tmpoctave2.append(k2.octave)
            tmpclassid2.append(k2.class_id)
        tmppt.append(tmppt2)
        tmpsize.append(tmpsize2)
        tmpangle.append(tmpangle2)
        tmpresponse.append(tmpresponse2)
        tmpoctave.append(tmpoctave2)
        tmpclassid.append(tmpclassid2)
        
   
    tmp = (tmppt,tmpsize,tmpangle,tmpresponse,tmpoctave,tmpclassid)
    return  tmp

'''
'' Aquesta funcio restableix els keypoints desats a disk
'''
def convertirVarToKeypoints(tmp):
    keypoints = []
    tmppt = tmp[0]
    tmpsize = tmp[1]
    tmpangle = tmp[2]
    tmpresponse = tmp[3]
    tmpoctave = tmp[4]
    tmpclassid = tmp[5]
    for i in range(len(tmppt)):
        tmppt2 = tmppt[i]
        tmpsize2 = tmpsize[i]
        tmpangle2 = tmpangle[i]
        tmpresponse2 = tmpresponse[i]
        tmpoctave2 = tmpoctave[i]
        tmpclassid2 = tmpclassid[i]
        ks = []
        for j in range(len(tmppt2)):
            k = cv2.KeyPoint(x=tmppt2[j][0],y=tmppt2[j][1],_size=tmpsize2[j],
                             _angle=tmpangle2[j],_response=tmpresponse2[j],
                             _octave=tmpoctave2[j],_class_id=tmpclassid2[j])
            ks.append(k)
        keypoints.append(ks)
    return keypoints



if __name__ == '__main__':
    pass

Models=[]
finalDescriptorsClasse = [];
finalKeypointsClasse = [];
flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
orbdetector = cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)

folders=sorted(glob.glob('..\\..\\..\\..\\ICAR\\train\\*'))
Models = []
classDescriptors = []
classKeypoints = []

if os.path.isfile('..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\objs.pickle'):
    with open('..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\objs.pickle','rb') as f:
        classDescriptors,tmp,Models = pickle.load(f)
    # carreguem els keypoints
    classKeypoints = convertirVarToKeypoints(tmp)
else :

    '''TRAIN'''
    
    # fer-ho per a cada carpeta
    for f in folders:
        print f
        L=Docmodel(f)
        Models.append(f)
        classDescriptors.append(L.getDescriptors())
        classKeypoints.append(L.getKeypoints())
    
    tmpkeyp = convertirKeypointsToVars(classKeypoints)
        
    # store the descriptors, keypoints on disk
    with open('..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\objs.pickle', 'wb') as f:
        pickle.dump((classDescriptors,tmpkeyp,Models), f)    

'''TEST'''
    
improc=0.0
contok=0.0
fileout=open('..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\confmatORB.dat','w')
folders_test=sorted(glob.glob('..\\..\\..\\..\\ICAR\\Mobile2\\*'))
#folders=sorted(glob.glob('..\\..\\..\\..\\ICAR\\test\\*'))
times=[]


aux2 = []
matcher = cv2.FlannBasedMatcher(flann_params, {})
for m in classDescriptors:
    matcher.add([np.asarray(m)]);

for f in folders_test:
    d=glob.glob(f+'\\*.jpg')
    for d2 in d:
        t=time.time()
        improc+=1
        img=cv2.imread(d2)   
        img = cv2.pyrDown(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orbdetector.detectAndCompute(gray,None)
        if descriptors!= None:
            matches = matcher.knnMatch(descriptors, k = 2)
            matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75 or m[0].imgIdx==m[1].imgIdx)]
        if len(matches)>0:
            immatches = np.array([m.imgIdx for m in matches])
            trainmatchesId = np.array([m.trainIdx for m in matches])
            querymatchesId = np.array([m.queryIdx for m in matches])
            c = np.bincount(np.array([m.imgIdx for m in matches]))
            #interestc = (c>10).nonzero()[0]
            interestc = np.argsort(c)[::-1]
            #interestc = interestc[0:10]
            if len(interestc)>0:
                votes = np.zeros(len(interestc),np.uint16)
                # comencem RANSAC
                for j in range(len(interestc)):
                    cx = interestc[j]
                    if c[cx]>4:
                        kptrainpt = []
                        kpquerypt = []
                        # agafem els descriptors d'interes
                        for i in range(len(trainmatchesId)):
                            if immatches[i] == cx:
                                kptrainpt.append(classKeypoints[cx][trainmatchesId[i]].pt)
                                kpquerypt.append(keypoints[querymatchesId[i]].pt)
                        # calculem la homografia
                        hom,mask = cv2.findHomography(np.float32(kptrainpt),np.float32(kpquerypt),cv2.RANSAC,5.0)
                        # comprobem que el RANSAC hagi trobat una homografia
                        if not mask is None:
                            # apuntem aquells descriptors que realment mostren una coherencia espaial
                            votes[j] = np.sum(mask.reshape(-1))
                    # ens quedem amb el maxim
                    cn = interestc[np.argmax(votes)]
                    d = interestc[np.argsort(votes)[::-1]] 
            else:
                cn = np.argmax(c)
                d = np.argsort(c)[::-1]
            doc=Models[cn]
            fileout.write(d2.split('\\')[3]+','+doc.split('\\')[3]+'\n')
            print d2.split('\\')[3]+','+doc.split('\\')[3]
            # calcular L1
            if(d2.split('\\')[3]==doc.split('\\')[3]):
            # calcular L>1
            #for x in d[0:10]:
                #if Models[x].split('\\')[3] == d2.split('\\')[3]:
                    contok+=1
                    break
        print 'ORB ' + str(contok*100/improc) + ' '+ str(time.time()-t)
        times.append(time.time()-t)
fileout.close()

print np.mean(times)
    
f=open('..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\confmatORB.dat')
data=f.readlines()
f.close()

y_pred=[]
y_test=[]
for i in data:
    y_test.append(i.split(',')[0])
    y_pred.append(i.split(',')[1].split('\n')[0])
cm = confusion_matrix(y_test, y_pred)
cm2=100*cm/np.sum(cm,axis=1)

# save the confusion matrix
np.savetxt("..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\matrix.txt", np.asarray(cm2,np.uint32), fmt='%d')

# write the list of test labels
f=open("..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\test.txt",'w')
for y in y_test :
    f.write(y+"\n");
f.close();
    
# write the list of pred labels
f=open("..\\..\\..\\..\\ICAR\\Results\\Classificacio_v4\\pred.txt",'w')
for y in y_pred :
    f.write(y+"\n");
f.close();
    


plt.matshow(cm2)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



    

        