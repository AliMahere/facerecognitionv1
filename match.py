import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


def face_distance(known_face_encodings, face_encoding_to_check):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(known_face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)


def find_cosine_distance( known_face_encodings, face_encoding_to_check):
    face_encoding_to_check =  np.array([face_encoding_to_check])
    known_face_encodings =  np.array(known_face_encodings)
    # num=np.dot(known_face_encodings ,face_encoding_to_check.T)
    # p1=np.sqrt(np.sum(known_face_encodings**2,axis=1))[:,np.newaxis]
    # p2=np.sqrt(np.sum(face_encoding_to_check**2,axis=1))[np.newaxis,:]
    # result = num/(p1*p2)
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
        
        f = find_cosine_distance( known_face_encodings, face_encoding_to_check)

        return list(f >= threshold)
    elif similarty == 'euclidean':
        f = face_distance(known_face_encodings, face_encoding_to_check)
        return list(f <= threshold)
    else:
        f = face_distance(known_face_encodings, face_encoding_to_check)
        return list(f <= threshold)
    