import numpy as np
import math
from math import sqrt
from math import inf

def euclidean_dist(a,b):
    acumulado=0.0
    for i in range (0,len(a)):
        acumulado+=(a[i]-b[i])**2
    return sqrt(acumulado)

def longest_patterns(data):
    max_value=-float(inf)
    patterns=[]
    for i in data:
        #print (i)
        max_value=max(max_value,i[0])
    if max_value==-inf:
        return None
    for i in data:
        if (i[0]==max_value):
            patterns.append(i)
    return patterns

def no_overlap(data):
    indeces=[]
    for i in data:
        for j in range (i[1][0],i[2][0]+1):
            if j not in indeces:
                indeces.append(j)
        for j in range (i[1][1],i[2][1]+1):
            if j not in indeces:
                indeces.append(j)
    return indeces

#error =0.1 --> 90% of similarity
def get_indeces(data,error=0.1):
    similarity_matrix=[]
    repeated_patterns=[]
    accepted_value=1-error
    for i in data:
        row=[]
        for j in data:
            row.append(1/(1+euclidean_dist(i,j)))
        similarity_matrix.append(row)
    for i in range(0,len(similarity_matrix)):
        for j in range(0,i):

            if(similarity_matrix[i][j]>=accepted_value):
                start_point=[i,j]
                length=0
                i_idx=i
                j_idx=j
                while(i_idx<len(similarity_matrix) and j_idx<len(similarity_matrix)):
                    if(similarity_matrix[i_idx][j_idx]<accepted_value):
                        break
                    else:
                        end_point=[i_idx,j_idx]
                        length+=1
                        i_idx+=1
                        j_idx+=1
                row=[length,start_point,end_point]
                repeated_patterns.append(row)
    repeated_patterns=longest_patterns(repeated_patterns)
    if repeated_patterns==None:
        return None
    indeces=no_overlap(repeated_patterns)
    return indeces
