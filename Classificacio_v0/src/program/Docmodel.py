import cv2
import numpy as np

class Docmodel(object):
    __slots__ = ['filename','img','gray','desc','keyp','num','thumb','quad']
    def __init__ (self, filename,dete, descr):
    	self.filename=filename.replace('./models/','')
        self.img=cv2.pyrDown(cv2.imread(filename))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thumb = cv2.resize(self.img,(100,100))
        self.quad = np.float32([[1, 1], [self.img.shape[1], 1], [self.img.shape[1], self.img.shape[0]], [1, self.img.shape[0]]])
        detector = cv2.FeatureDetector_create(dete)
        descriptor = cv2.DescriptorExtractor_create(descr)
        self.keyp = detector.detect(self.gray)
        self.keyp, self.desc = descriptor.compute(self.gray,self.keyp)
        self.num=len(self.keyp)
        
    def getImg(self):
        return self.img
    def getFilename(self):
        return self.filename
    def getGray(self):
        return self.gray
    def getKeypoints(self):
        return self.keyp
    def getDescriptors(self):
        return self.desc
    def getNum(self):
        return self.num
    def getThumb(self):
        return self.thumb
    def getQuad(self):
        return self.quad
		
		

