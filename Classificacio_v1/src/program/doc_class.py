import cv2
import glob
import numpy as np
from Docmodel import *
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import os


orbdetector=cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)


Models=[]

flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
matcher = cv2.FlannBasedMatcher(flann_params, {})


folders=sorted(glob.glob('D:\\ICAR\\train\\*'))
for f in folders:
    d=glob.glob(f+'\\*.jpg')
    print d[0]
    L=Docmodel(d[0])
    Ckeyp=L.getKeypoints()
    Cdesc=L.getDescriptors()
    matchertmp = cv2.FlannBasedMatcher(flann_params, {})
    matchertmp.add([Cdesc])
    votes=[0]*len(Ckeyp)
    cont=0.0
    for trains in d[1:]:
        cont+=1
        imgtrain=cv2.pyrDown(cv2.imread(trains))
        graytrain = cv2.cvtColor(imgtrain, cv2.COLOR_BGR2GRAY)
        (keyptrain,desctrain)=orbdetector.detectAndCompute(graytrain,None)
        matches = matchertmp.knnMatch(desctrain, k = 3)
        matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75)]
        for m in range(len(matches)):
            votes[matches[m].trainIdx]+=1
    
    #idx=[x[0] for x in enumerate(map(lambda x: x\\cont, votes)) if x[1]<0.1]
    idx=[i[0] for i in sorted(enumerate([v/cont for v in votes]), key=lambda x:x[1])][0:len(votes)-100]
    
    for index in sorted(idx,reverse=True):
        del Ckeyp[index]
        del votes[index]
        Cdesc=np.delete(Cdesc,index,0)
    L.desc=Cdesc
    L.keyp=Ckeyp
    print len(Ckeyp)
    matcher.add([L.getDescriptors()])
    Models.append(L)

improc=0.0
contok=0.0
fileout=open('D:\\ICAR\\Results\\Classificacio\\confmatORB.dat','w')
folders=sorted(glob.glob('D:\\ICAR\\test\\*'))
times=[]
for f in folders:
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
            c = np.bincount(np.array([m.imgIdx for m in matches]))	
            doc=Models[np.argmax(c)]
            fileout.write(d2.split('\\')[3]+','+doc.filename.split('\\')[3]+'\n')
            l=sorted(range(len(c)), key=lambda k: c[k],reverse=True)
            if(d2.split('\\')[3]==doc.filename.split('\\')[3]):
            #if [Models[x].filename.split('\\')[3] for x in l[0:10]].__contains__(d2.split('\\')[3]):
                contok+=1
        print 'ORB ' + str(contok*100/improc) + ' '+ str(time.time()-t)
        times.append(time.time()-t)
fileout.close()

print np.mean(times)
    
f=open('D:\\ICAR\\Results\\Classificacio\\confmatORB.dat')
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
np.savetxt("D:\\ICAR\\Results\\Classificacio\\matrix.txt", np.asarray(cm2,np.uint32), fmt='%d')

# write the list of test labels
f=open("D:\\ICAR\\Results\\Classificacio\\test.txt",'w')
for y in y_test :
    f.write(y+"\n");
f.close();
    
# write the list of pred labels
f=open("D:\\ICAR\\Results\\Classificacio\\pred.txt",'w')
for y in y_pred :
    f.write(y+"\n");
f.close();
    


plt.matshow(cm2)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()