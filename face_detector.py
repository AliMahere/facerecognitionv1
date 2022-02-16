import cv2

class FaceDetector():

    def __init__(self, device = 'cpu', detector = 0, conf_threshold = 0.6,
                config_file = "models/deploy.prototxt", 
                model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"):
        """FaceDetctor class used to detect faces insde a frame .
        Args:
            device (int, optional): [description]. Defaults to 0.
        """
        self.config_file =config_file
        self.model_file= model_file
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.device = device
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        #self.net = self.set_model()
        coun = 0 



    def detectFaceOpenCVDnn(self, frame):
        """[summary]

        Args:
            net (cv2.dnn_Net): A CNN model used for face detection Default (res10)
            frame (cv2.MAT): an image contains faces 
            framework (int, optional): selected framrework for detect 0 for caffe , 1 for tensorflow . Defaults to 0.
            conf_threshold (float, optional): a thrishold of the minium likelihood of the detected faces. Defaults to 0.7.

        Returns:
            frameOpencvDnn (cv2.MAT): a cv2.MAT has original farme and drown boxes around detected faces .
            bboxes (list) : a 2d list of size (number_of_faces, 4 ) constains startX, startY, endX, endY.
        """
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        if self.detector == 0:
            blob = cv2.dnn.blobFromImage(
                frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
            )
        else:
            blob = cv2.dnn.blobFromImage(
                frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False,
            )

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(
                    frameOpencvDnn,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    int(round(frameHeight / 150)),
                    8,)
        return frameOpencvDnn, bboxes

    def set_model(self):
        """
        set another model rather than defualt for face detector 

        Args:
            model_file (file): a file contains model weihts
            config_file (file):  a file contains model structre 
            framework (int): set framework of model 0 for caffe 1 for tensorflow or use FaceDetector.FRAMEWORK_CAFFE/FaceDetector.FRAMEWORK_TF
        """

        if self.detector == 0:
            self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
        else:
            self.net = cv2.dnn.readNetFromTensorflow(self.config_file, self.model_file)

        if self.device == 'cpu':
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
    def set_device(self, device):
        """set device used for processign default CPU

        Args:
            device (int): 0 for CPU 1 for GPU or use FaceDetector.CPU/FaceDetector.GPU
        """
        self.device = device
        if device == 'cpu':
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def set_conf_threshold(self, threshold):
        """set thrishold of the minium likelihood of the detected faces. Defaults to 0.7.

        Args:
            threshold (float): thrishold of the minium likelihood
        """
        self.conf_threshold = threshold

    def set_framework(self, framework):
        """
            set framrework for detection model 0 for caffe , 1 for tensorflow . Defaults to 0
        Args:
            framework (float): use FaceDetector.FRAMEWORK_CAFFE/FaceDetector.FRAMEWORK_TF
        """
        self.detector = framework

    def detect_frame(self, frame):
        """pass a frame 

        Args:
            frame (cv2.Mat): an image contains faces 

        Returns:
            frameOpencvDnn (cv2.MAT): a cv2.MAT has original farme and drown boxes around detected faces .
            bboxes (list) : a 2d list of size (number_of_faces, 4 ) constains startX, startY, endX, endY.
        """
        outOpencvDnn, bboxes = self.detectFaceOpenCVDnn( frame)
        return  outOpencvDnn ,bboxes 




