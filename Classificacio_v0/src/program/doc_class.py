import cv2
import glob
import numpy as np
from Docmodel import *
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


dtct="ORB"
dscrpt="ORB"

detector = cv2.FeatureDetector_create(dtct)
descriptor = cv2.DescriptorExtractor_create(dscrpt)



Models=[]

if(dscrpt=="SIFT"):
	flann_params= dict(algorithm = 0,trees=5)
	matcher = cv2.FlannBasedMatcher(flann_params, dict(checks=50))
else:
	flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
	matcher = cv2.FlannBasedMatcher(flann_params, {})



folders=sorted(glob.glob('D:\\Icar\\train\\*'))
for f in folders:
    d=glob.glob(f+'\\*.jpg')
    print d[0]
    L=Docmodel(d[0],dtct,dscrpt)
    matcher.add([L.getDescriptors()])
    Models.append(L)

improc=0.0
contok=0.0
fileout=open('D:\\Icar\\Results\\Classificacio\\confmatORB.dat','w')
folders=sorted(glob.glob('D:\\Icar\\test\\*'))
times=[]
for f in folders:
    d=glob.glob(f+'\\*.jpg')
    for d2 in d:
        t=time.time()
        improc+=1
        img=cv2.imread(d2)   
        img = cv2.pyrDown(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        #print len(keypoints)
        keypoints, descriptors = descriptor.compute(gray,keypoints)
        if descriptors!= None:
            matches = matcher.knnMatch(descriptors, k = 2)
            matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75 or m[0].imgIdx==m[1].imgIdx)]
        if len(matches)>0:
            c = np.bincount(np.array([m.imgIdx for m in matches]))	
            doc=Models[np.argmax(c)]
            fileout.write(d2.split('\\')[3]+','+doc.filename.split('\\')[3]+'\n')
            l=sorted(range(len(c)), key=lambda k: c[k],reverse=True)
            #if(d2.split('/')[2]==doc.filename.split('/')[2]):
            if [Models[x].filename.split('\\')[3] for x in l[0:10]].__contains__(d2.split('\\')[3]):
                contok+=1
        print 'ORB ' + str(contok*100/improc) + ' '+ str(time.time()-t)
        times.append(time.time()-t)
fileout.close()

print np.mean(times)
    
f=open('D:/Icar/Results/Classificacio/confmatORB.dat')
data=f.readlines()
f.close()

y_pred=[]
y_test=[]
for i in data:
    y_test.append(i.split(',')[0])
    y_pred.append(i.split(',')[1].split('\n')[0])
cm = confusion_matrix(y_test, y_pred)
cm2=100*cm/np.sum(cm,axis=1)
plt.matshow(cm2)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()