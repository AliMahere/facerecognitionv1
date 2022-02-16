import cv2
from vgg import loadModelVGG 
import tensorflow as tf
from uti import find_input_shape, prepareInput
import numpy as np

class FaceEmbder():

    def __init__(self, device = 'cpu' ,model_name = 'openface' ):
        """FaceDetctor class used to detect faces insde a frame .
        Args:
            device (int, optional): [description]. Defaults to cpu.
        """
        self.device = device
        self.model_name = model_name
        self.net = self.set_model()
        if model_name ==  'vgg':
            self.target_shape = find_input_shape(self.net)



    def get_embdings(self, faceImg):
        if self.model_name =='vgg':
            input_shape_x, input_shape_y = find_input_shape(self.net)
            inp = prepareInput(faceImg, (input_shape_x, input_shape_y))
            embedding = self.net.predict(inp)[0].tolist()
            return embedding

        elif self.model_name =='openface':

            faceBlob = cv2.dnn.blobFromImage(faceImg, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(faceBlob)
            vec = self.net.forward()
            return vec.flatten()


	#------------------------------------------
    def set_model(self):
        """
        set another model rather than defualt for face detector 

        Args:
            model_file (file): a file contains model weihts
            config_file (file):  a file contains model structre 
            framework (int): set framework of model 0 for caffe 1 for tensorflow or use FaceDetector.FRAMEWORK_CAFFE/FaceDetector.FRAMEWORK_TF
        """
        if self.model_name == 'vgg':
            #TODO:chech device
            return loadModelVGG()
        elif self.model_name == 'openface':
            if self.device == 'cpu':
                path = 'models/openface.nn4.small2.v1.t7'
                model = cv2.dnn.readNetFromTorch(path)
                model.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
                return model


    def set_device(self, device):
        pass
    



