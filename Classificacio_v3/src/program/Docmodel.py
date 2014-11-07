'''
Created on 27/10/2014

@author: lluispere
'''


import cv2
import numpy as np
import glob

class Docmodel(object):
    #__slots__ = ['foldername','maxdesc','desc_finals','desc_totals','references','self.votestf','self.votesidf','orb','matcher_params']
    #def __init__ (self, filename,dete, descr):
    def __init__ (self, foldername):
        self.orb=cv2.ORB(5000,scaleFactor=1.2, nlevels=8, edgeThreshold=5, firstLevel=0, WTA_K=2, scoreType=1, patchSize=31)
        self.foldername=foldername.replace('./models/','')
        self.maxdesc = 100
        self.desc_finals = [] # descriptors representatius del model
        self.desc_imatges = [] # descriptors de cada imatge
        self.desc_totals = [] # tots els descriptors del model
        self.matcher_params= dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1)
        
        # llegir les imatges i calcular descriptors
        d=glob.glob(foldername+'\\*.jpg')
        cont = 0
        llista = []
        
        # calculem tots els descriptors de les imatges del model
        for f in d:
            img=cv2.pyrDown(cv2.imread(f))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (_,descriptors)=self.orb.detectAndCompute(gray,None)
            for descAux in descriptors: self.desc_totals.append(descAux)
            self.desc_imatges.append(descriptors)
            llista.append(cont)
            cont += len(descriptors)
            
        # inicialitzem el nombre de vots als descriptors de la classe aixi com les referencies creuades    
        self.votestf = np.zeros((cont,1),np.uint32)
        self.votesidf = np.zeros((cont,1),np.uint32)
        self.references = [[]]*cont
        
        # fer el matching dels orb 2 a 2
        i = 0
        for desc in self.desc_imatges:
            j = 0
            for desc2 in self.desc_imatges:
                if i == j:
                    j+=1
                    continue
                    
                else:
                    matcher = []
                    matcher = cv2.FlannBasedMatcher(self.matcher_params, {})
                    matcher.add([np.asarray(desc)])
                    matches = matcher.knnMatch(np.asanyarray(desc2), k = 2)
                    matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75)]
                    # desem els matches a la llista de vots i a les referencies creuades
                    for m in matches:
                        self.votestf[llista[i]+m.trainIdx]+=1
                        a = list(self.references[llista[i]+m.trainIdx])
                        a.append(llista[j]+m.queryIdx)
                        self.references[llista[i]+m.trainIdx]=a
                j+=1
            i+=1              
        
        
    '''
    '' Aquesta funcio fa matching entre els descriptors del model i els que son passats com a argument.
    '' A mes estora en votesidf els matchings
    '''        
    def compareWithIdf(self,descriptors):
        matchertmp = cv2.FlannBasedMatcher(self.matcher_params, {})
        matchertmp.add([np.asarray(self.desc_totals)])
        matches = matchertmp.knnMatch(np.asarray(descriptors), k = 2)
        matches = [m[0] for m in matches if len(m) >= 2 and (m[0].distance < m[1].distance * 0.75)]
        for m in matches:
            self.votesidf[m.trainIdx]+=1

    '''
    '' Aquesta funcio calcula els descriptors mes representatius de la classe tenint en compte el tf-idf 
    '''            
    def calculateBestTfIdfDesctiptors(self, number_of_classes):
 
        # normalitzacio de vots idf
        norm_idf = np.log((1+np.max(self.votesidf))/(1+self.votesidf))
        # calcul de vots conjunt tf-idf
        finalvotes = self.votestf*norm_idf
        
        # ara hem d'escollir els descriptors aquells que tenen mes vots
        for _ in range(self.maxdesc):
            idxOrdenats = np.argsort(finalvotes,axis=0)[::-1]
            maxim = idxOrdenats[0]
            self.desc_finals.append(self.desc_totals[maxim])
            for x1 in self.references[maxim]:
                finalvotes[x1]=0
            finalvotes[maxim]=0

        
    def getFoldername(self):
        return self.foldername
    def getDescriptorsFinals(self):
        return self.desc_finals
    def getDescriptorsImatges(self):
        return self.desc_imatges
    def getDescriptorsTotals(self):
        return self.desc_totals
    def getVotesTf(self):
        return self.votestf
    def getVotesIdf(self):
        return self.votesidf
    def getReferences(self):
        return self.references
    def getOrb(self):
        return self.orb
    def getMaxDesc(self):
        return self.maxdesc
        
