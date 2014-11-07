'''
Created on 27/10/2014

@author: lluispere
'''


import cv2
import numpy as np
from pydoc import describe
import glob

class Docmodel2(object):
    __slots__ = ['foldername','maxdesc','desc','orb']
    #def __init__ (self, filename,dete, descr):
    def __init__ (self, foldername):
        self.orb=cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)
        self.foldername=foldername.replace('./models/','')
        self.maxdesc = 100
        self.desc = []
        
        flann_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
        
        # llegir les imatges i calcular descriptors
        d=glob.glob(foldername+'\\*.jpg')
        cont = 0
        llista = []
        descLlistaAux = []
        
        for f in d:
            img=cv2.pyrDown(cv2.imread(f))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (kp,descriptors)=self.orb.detectAndCompute(gray,None)
            for descAux in descriptors: descLlistaAux.append(descAux)
            self.desc.append(descriptors)
            llista.append(cont)
            cont += len(descriptors)
            
            
        votes = np.zeros((cont,1),np.uint32)
        ref = [[]]*cont
        
        # fer el matching dels orb 2 a 2
        i = 0
        for desc in self.desc:
            j = 0
            for desc2 in self.desc:
                if i == j:
                    j+=1
                    continue
                    
                else:
                    matcher = []
                    matcher = cv2.FlannBasedMatcher(flann_params, {})
                    matcher.add([np.asarray(desc)])
                    matches = matcher.knnMatch(np.asanyarray(desc2), k = 2)
                    matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75)]
                    for m in matches:
                        votes[llista[i]+m.trainIdx]+=1
                        a = list(ref[llista[i]+m.trainIdx])
                        a.append(llista[j]+m.queryIdx)
                        ref[llista[i]+m.trainIdx]=a
                j+=1
            i+=1
        # ara hem d'escollir els descriptors aquells que tenen mes vots
        self.desc = []
        for x in range(self.maxdesc):
            idxOrdenats = np.argsort(votes,axis=0)[::-1]
            maxim = idxOrdenats[0]
            self.desc.append(descLlistaAux[maxim])
            for x1 in ref[maxim]:
                votes[x1]=0
            votes[maxim]=0
                         
        
        
        
        
    def getFoldername(self):
        return self.foldername
    def getDescriptors(self):
        return self.desc
    def getOrb(self):
        return self.orb
    def getMaxDesc(self):
        return self.maxdesc
        
