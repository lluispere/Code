'''
Created on 26/11/2014

@author: lpheras
'''

import os   

if __name__ == '__main__':
    pass

input_path = "D:\\ICAR\\GroundTruth\\original\\";
file_path_name = "D:\\ICAR\\GroundTruth\\original\\files.txt"

f = open(file_path_name,'w');

for fo in os.listdir(input_path):
    f.write(input_path + fo + "\n");
    
f.close;