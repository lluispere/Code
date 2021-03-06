'''
Created on Oct 24, 2014

@author: lpheras
'''
import os
import numpy as np
from shutil import copyfile


if __name__ == '__main__':
    pass

matrixPath = "..\\..\\..\\..\\Results\\Classificacio_v5\\test500\\matrix.txt";
testPath = "..\\..\\..\\..\\test\\";
trainPath = "..\\..\\..\\..\\train\\";
outPath = "..\\..\\..\\..\\Results\\Classificacio_v5\\test500\\Confusions\\";
outPathFiles = "..\\..\\..\\..\\Results\\Classificacio_v5\\test500\\";

# check whether the matrix exists
if os.path.isfile(matrixPath) :
    matrix = [];
    # read file into a list
    with open(matrixPath) as f :
        for fila in f.read().splitlines() :
            llista = []
            for x in fila.split(' ') :
                llista.append(int(x));
            matrix.append(llista)
    # en aquest punt tenim la matriu llegida        
    matrix = np.asarray(matrix);
    
    # mirar quines s'equivoca
    i = 0;
    file_classe_resutls=open(outPathFiles + "class_results.txt",'w')
    votes = np.zeros(matrix.shape[0],np.uint64)
    votes_correctes = np.zeros(matrix.shape[0],np.uint64)
    llistaClasses = os.listdir(testPath)
    for d in llistaClasses :
        num_imatges = len(os.listdir(testPath+d))
        # si la classificacio que li correspon es menor al 50
        fila = matrix[i]
        votes_correctes[i] = fila[i]
        file_classe_resutls.write(d + " " + str(fila[i]))
        v = np.max(fila);
        votes_error = fila*num_imatges/100
        votes_error[i] = 0
        votes += votes_error
        iMax = np.argmax(fila);
        # si el maxim no correspon al que pertoca
        # i el maxim es major al 30%
        if (iMax != i):
            file_classe_resutls.write(" " + str(llistaClasses[iMax]) + " " + str(fila[iMax]) + "\n")
            # crear la carpeta corresponent
            d2 = llistaClasses[iMax];
            nomCarpeta = outPath+d+"-"+d2;
            if not os.path.isdir(nomCarpeta) :
                os.makedirs(nomCarpeta);
            # desar dues instancies de cada classe
            im1Names = os.listdir(trainPath+d);
            im2Names = os.listdir(testPath+d2);
            for m in range(2) :
                copyfile(trainPath+d+"\\"+im1Names[m],nomCarpeta+"\\"+im1Names[m]);
                copyfile(testPath+d2+"\\"+im2Names[m],nomCarpeta+"\\"+im2Names[m]);
        else:
            file_classe_resutls.write("\n")
        i=i+1;

    print np.mean(votes_correctes)
    # ara valorem quines son les classes que mes capten vots erronis        
    file_classe_errors=open(outPathFiles + "class_errors.txt",'w')
    i = 0
    for d in llistaClasses :
        file_classe_errors.write(d+" "+str(votes[i])+"\n")
        i+=1
        
             
                
            
        
    
    
    

