'''
Created on Oct 6, 2014

@author: lpheras
'''
# primer programa
import os
import csv
from shutil import copyfile



# path with the csvs
csvPathTrain = "D:/ICAR/GroundTruth/train.csv"
csvPathTest = "D:/ICAR/GroundTruth/test.csv"
# path with the images 
imagesPathTrain = "D:/ICAR/GroundTruth/cropped/"
imagesPathTest = "D:/ICAR/GroundTruth/original/"
trainFolder = "D:/ICAR/GroundTruth/train"
testFolder = "D:/ICAR/GroundTruth/test"

#os.makedirs(trainFolder)
#os.makedirs(testFolder)

# parse the csv train
with open(csvPathTrain, "rb") as csvFile:
    a = csv.reader(csvFile, delimiter=";")
    
    cont = 0
    
    # store the classes and the images
    for row in a:
        
        # if it is not the first row
        if cont <> 0:
            imageName = row[0]
            gtName = row[1]
            imageSourcePath = imagesPathTrain + "/" + imageName
            classDestinationPath = trainFolder + "/" + gtName
            imageDestinationPath = trainFolder + "/" + gtName + "/" + imageName
                                    
            # create the path if it does not exist
            if not os.path.exists(classDestinationPath) :
                os.makedirs(classDestinationPath)
            
            # copy the image into the new path
            if os.path.isfile(imageSourcePath) :
                copyfile(imageSourcePath,imageDestinationPath)
            else :
                print imageSourcePath
        
        cont = 1
        
        
# parse the csv test
with open(csvPathTest, "rb") as csvFile:
    a = csv.reader(csvFile, delimiter=";")
    
    cont = 0
    
    # store the classes and the images
    for row in a:
        
        # if it is not the first row
        if cont <> 0:
            imageName = row[0]
            gtName = row[1]
            imageSourcePath = imagesPathTest + "/" + imageName
            classDestinationPath = testFolder + "/" + gtName
            imageDestinationPath = testFolder + "/" + gtName + "/" + imageName
                                    
            # create the path if it does not exist
            if not os.path.exists(classDestinationPath) :
                os.makedirs(classDestinationPath)
            
            # copy the image into the new path
            if os.path.isfile(imageSourcePath) :
                copyfile(imageSourcePath,imageDestinationPath)
            else :
                print imageSourcePath
        
        cont = 1

