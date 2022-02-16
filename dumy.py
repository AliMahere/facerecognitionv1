import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import cv2
from sklearn.metrics.pairwise import euclidean_distances

def find_cosine_distance( known_face_encodings, face_encoding_to_check):
    face_encoding_to_check =  np.array(face_encoding_to_check)

    num=np.dot(known_face_encodings ,face_encoding_to_check.T)
    p1=np.sqrt(np.sum(known_face_encodings**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(face_encoding_to_check**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def csm(A,B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)
def euclidean_distance( known_face_encodings, face_encoding_to_check):

    return euclidean_distances(known_face_encodings, face_encoding_to_check)


A = np.array(
[[0, 1, 0, 0, 1],
[0, 0, 1, 1, 1],
[1, 1, 0, 1, 0]])
B =  np.array(
[[0, 1, 0, 0, 1]])


# A=np.array([10,3])
# B=np.array([8,7])
result=euclidean_distance(A,B)
print(result)
