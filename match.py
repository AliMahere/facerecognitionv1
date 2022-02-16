import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


def find_face_euclidean_distance( known_face_encodings, face_encoding_to_check):

    face_encoding_to_check =  np.array([face_encoding_to_check])
    known_face_encodings =  np.array(known_face_encodings)
    return euclidean_distances(known_face_encodings, face_encoding_to_check)

#def find_face_cosine_distance( known_face_encodings, face_encoding_to_check):
    #face_encoding_to_check =  np.array([face_encoding_to_check])
    #known_face_encodings =  np.array(known_face_encodings)
    # num=np.dot(known_face_encodings ,face_encoding_to_check.T)
    # p1=np.sqrt(np.sum(known_face_encodings**2,axis=1))[:,np.newaxis]
    # p2=np.sqrt(np.sum(face_encoding_to_check**2,axis=1))[np.newaxis,:]
    # result = num/(p1*p2)

def find_face_cosine_distance( known_face_encodings, face_encoding_to_check):

    face_encoding_to_check =  np.array([face_encoding_to_check])
    known_face_encodings =  np.array(known_face_encodings)
    result=cosine_similarity(known_face_encodings,face_encoding_to_check)
    return result




def compare_faces(known_face_encodings, face_encoding_to_check, similarty = 'cos',threshold=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    if similarty == 'cos':
        
        f = find_face_cosine_distance( known_face_encodings, face_encoding_to_check)

        return list(f >= threshold)
    elif similarty == 'euclidean':
        f = find_face_euclidean_distance(known_face_encodings, face_encoding_to_check)
        return list(f <= threshold)
    else:
        f = find_face_euclidean_distance(known_face_encodings, face_encoding_to_check)
        return list(f <= threshold)
    