'''
Created on Oct 24, 2014

@author: lpheras
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

orbdetector=cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)
Models=[]
maxdescriptors = 500;
finalDescriptorsClasse = [];
finalKeypointsClasse = [];

flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
matcher = cv2.FlannBasedMatcher(flann_params, {})

folders=sorted(glob.glob('D:\\ICAR\\train\\*'))


if os.path.isfile('D:\\ICAR\\Results\\Classificacio2\\objs.pickle'):
    with open('D:\\ICAR\\Results\\Classificacio2\\objs.pickle','rb') as f:
        finalDescriptorsClasse,Models = pickle.load(f)
else :

    '''TRAIN'''
    
    # fer-ho per a cada carpeta
    for f in folders:

        print f
        d=glob.glob(f+'\\*.jpg')
        # creem el model 
        Models.append(d[0])
        descriptorsClasse = [];
        keypoints = [];
        descriptors = []
        
            
        # per a cada imatge
        for im in d :
            # calculem els desc i keyp de la imatge
            img=cv2.pyrDown(cv2.imread(im))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (keyp,desc)=orbdetector.detectAndCompute(gray,None)
                   
            for x in desc : descriptors.append(x);
            for x in keyp : keypoints.append(x);
                
        # creem el matcher i afegim tots els descriptors de la classe
        matchertmp = cv2.FlannBasedMatcher(flann_params, {})
        matchertmp.add([np.asanyarray(descriptors)]);
        
        # calculem els matches i ens quedem amb aquells que realment fan matching
        matches = matchertmp.knnMatch(np.asanyarray(descriptors), k = len(d))
        
        # ens quedem amb les distancies dels macthes
        distances = [];
        for match in matches:
            for m in match:
                distances.append(m.distance);
        
        # threshold sobre la distancia dels matches
        thr = np.mean(np.asarray(distances))/2;
        
        # apuntem els matches representatius i els guardem a votes
        finalmatches = [];
        for match in matches:
            for m in range(len(match)):
                if (m>0) & (match[m].distance < thr):
                    finalmatches.append(match[m])
        #matches = [m[1] for m in matches if len(m) >= 3 and (m[1].distance < m[2].distance * 0.75)]
        #matches = [m[1] for m in matches if len(m) >= 3 and (m[1].distance < m[2].distance * 0.75)]
        
        votes = np.zeros((len(finalmatches),len(finalmatches)),np.uint16)
        trainids = []
        queryids = [];
        for m in range(len(finalmatches)):
            trainids.append(finalmatches[m].trainIdx)
            queryids.append(finalmatches[m].queryIdx)
        
        trainids =  np.asarray(trainids)
        queryids = np.asarray(queryids)
            
        for m in range(len(finalmatches)):
                votes[(trainids==finalmatches[m].trainIdx).nonzero()[0][0]][(queryids==finalmatches[m].queryIdx).nonzero()[0][0]]+=1;
                #votes[matches[m].queryIdx][matches[m].trainIdx]+=1;
        
        md = maxdescriptors;
        # mentre no es superi el nombre maxim de descriptors per classe
        while md > 0:
            # mentre els votes siguin majors a 0
            suma = np.sum(votes, axis=1)
            # ordernar aquells que tenen mes vots i agafar el maxim
            sortedIdx = np.argsort(suma)[::-1]
            maxim = sortedIdx[0];
            maximaFila = votes[maxim];
            # aquells que han votat al descriptor han de decreixer
            nonzeros = (maximaFila>0).nonzero();
            for descId in nonzeros[0]:
                votes[descId][maxim] -= votes[maxim][descId] 
            # clear the descriptor
            votes[maxim] = 0;
            # store the relevant descriptor
            descriptorsClasse.append(descriptors[trainids[maxim]])
            
            md-=1
            
        finalDescriptorsClasse.append([descriptorsClasse])

    # store the descriptors, keypoints on disk
    with open('D:\\ICAR\\Results\\Classificacio2\\objs.pickle', 'wb') as f:
        pickle.dump((finalDescriptorsClasse,Models), f)
    

        

'''TEST'''
    
improc=0.0
contok=0.0
fileout=open('D:\\ICAR\\Results\\Classificacio\\confmatORB.dat','w')
folders=sorted(glob.glob('D:\\ICAR\\test\\*'))
times=[]


aux2 = []
# aixo nomes en el cas antic
for m in finalDescriptorsClasse:
    aux = [];
    for f1 in m[0]:
        aux.append(f1[0]);
    matcher.add([np.asarray(aux)]);
    
finalDescriptorsClasse=aux2;


# afegim els descriptors al matcher
#matcher.add([np.asanyarray(finalDescriptorsClasse)])

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