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

if __name__ == '__main__':
    pass

Models=[]
finalDescriptorsClasse = [];
finalKeypointsClasse = [];
flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
matcher = cv2.FlannBasedMatcher(flann_params, {})
orbdetector = cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)

Models = []
classDescriptors = [] 

# mirem si els descriptors son guardats a memoria
if os.path.isfile('D:\\ICAR\\Results\\Classificacio_v3\\objs.pickle'):
    with open('D:\\ICAR\\Results\\Classificacio_v3\\objs.pickle','rb') as f:
        classDescriptors,Models = pickle.load(f)
else :
    
    '''TRAIN'''
    # si no, comencem el training
    folders_train=sorted(glob.glob('D:\\ICAR\\train\\*'))
    
    # fer-ho per a cada carpeta
    for f in folders_train:
        print f
        L=Docmodel(f)
        cont = 0
        # ara cal comparar els descriptors de la classe contra les altres classes
        for f2 in folders_train:
            cont+=1
            if f == f2 :
                # si es tracta de la mateixa carpeta passem 
                continue
            LAux = Docmodel(f2)
            print "    vs "+f2
            L.compareWithIdf(LAux.getDescriptorsTotals())
        
        # un cop s'han comparat am tots els descriptors de les classes es moment de desar els descriptors finals
        L.calculateBestTfIdfDesctiptors(cont)
        Models.append(f)
        classDescriptors.append(L.getDescriptorsFinals())
        
    # store the descriptors, keypoints on disk
    with open('D:\\ICAR\\Results\\Classificacio_v3\\objs.pickle', 'wb') as f:
        pickle.dump((classDescriptors,Models), f)    

'''TEST'''
    
improc=0.0
contok=0.0
fileout=open('D:\\ICAR\\Results\\Classificacio_v3\\confmatORB.dat','w')
folders_test=sorted(glob.glob('D:\\ICAR\\test\\*'))
times=[]


aux2 = []
# aixo nomes en el cas antic
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
            c = np.bincount(np.array([m.imgIdx for m in matches]))    
            doc=Models[np.argmax(c)]
            fileout.write(d2.split('\\')[3]+','+doc.split('\\')[3]+'\n')
            print d2.split('\\')[3]+','+doc.split('\\')[3]
            l=sorted(range(len(c)), key=lambda k: c[k],reverse=True)
            if(d2.split('\\')[3]==doc.split('\\')[3]):
            #if [Models[x].filename.split('\\')[3] for x in l[0:10]].__contains__(d2.split('\\')[3]):
                contok+=1
        print 'ORB ' + str(contok*100/improc) + ' '+ str(time.time()-t)
        times.append(time.time()-t)
fileout.close()

print np.mean(times)
    
f=open('D:\\ICAR\\Results\\Classificacio_v3\\confmatORB.dat')
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
np.savetxt("D:\\ICAR\\Results\\Classificacio_v3\\matrix.txt", np.asarray(cm2,np.uint32), fmt='%d')

# write the list of test labels
f=open("D:\\ICAR\\Results\\Classificacio_v3\\test.txt",'w')
for y in y_test :
    f.write(y+"\n");
f.close();
    
# write the list of pred labels
f=open("D:\\ICAR\\Results\\Classificacio_v3\\pred.txt",'w')
for y in y_pred :
    f.write(y+"\n");
f.close();
    


plt.matshow(cm2)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

        