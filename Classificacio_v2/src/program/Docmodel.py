import cv2
import numpy as np

class Docmodel(object):
    __slots__ = ['filename']
    #def __init__ (self, filename,dete, descr):
    def __init__ (self, filename):
        self.filename=filename.replace('./models/','');
        
        
    def getFilename(self):
        return self.filename;
        

