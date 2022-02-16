import Augmentor
import os
from face_detector import FaceDetector
from load_dataset import LoadImages
from pathlib import Path
from random import randint
import cv2
import shutil
from opt import get_data_prepration_opt
import dlib
from imutils.face_utils.facealigner import FaceAligner



def xyxy2xywh(X):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    x = X[0]
    y = X[1]  
    w = X[2] - X[0]  # width
    h = X[3] - X[1]  # height
    return (x,y,w,h)


def balance_dataset():

    source, target_num_of_spamples =  opt.dataset, opt.spamples
    for subdir, dirs, files in os.walk(source):
        if len(files) == 0:
            continue
        p = str(Path(subdir).absolute())
        # save_dir =os.path.join(save_path ,p.split(os.path.sep)[-1])
        # if not os.path.exists(save_dir):
        #     print("false")
        #     os.makedirs(save_dir)
        print("_________len____",len(files))

        p = Augmentor.Pipeline(subdir)

        #p.rotate(probability=0.2, max_left_rotation=rotate_degree, max_right_rotation=rotate_degree)
        #p.zoom(probability=0.3, min_factor=1.1, max_factor=1.3)
        p.flip_left_right(probability=0.4)
        p.crop_random(probability=0.6 , percentage_area = 0.8)

        n_samples = target_num_of_spamples - len(files)
        if n_samples <= 0  :
            continue
        p.sample(n_samples)

        source_dir = os.path.join(subdir ,"output")
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), subdir)
        try: 
            os.remove(source_dir) 
        except OSError as error: 
            print(error) 
                

def create_faces_dataset():

    source, save_path =  opt.dataset, opt.save_path
    detector, predictor ,fa =  None , None, None

    if opt.aline :
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceWidth=256, desiredLeftEye = (0.25,0.25))

    #TODO: add to opt opt.device opt.confidence
    for subdir, dirs, files in os.walk(source):
        p = str(Path(subdir).absolute())
        save_dir =os.path.join(save_path ,p.split(os.path.sep)[-1] )
        p = str(Path(save_dir).absolute())
        fd = FaceDetector(device = opt.device, conf_threshold = opt.confidence)
        dataset = LoadImages(subdir)
        for path, img, _ in dataset:
            

            # ______check padding _________
            if opt.padding_ratio != 0:
                top = int(opt.padding_ratio * img.shape[0])  # shape[0] = rows
                bottom = top
                left = int(opt.padding_ratio * img.shape[1])  # shape[1] = cols
                right = left
                value = [randint(0, 255), randint(0, 255), randint(0, 255)]
                img = cv2.copyMakeBorder(img, top, bottom, left, right,  cv2.BORDER_CONSTANT, None, value)
            # _______________check padding ___________


            img_name = path.split(os.path.sep)[-1]
            img_save_path = os.path.join(p, img_name)
            _ , bboxes =  fd.detect_frame(img)


            if opt.aline :
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                for i in range(len(bboxes)):
                    faceimage = img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2]]
                    (fH, fW) = faceimage.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    dlibRect = dlib.rectangle(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]) 

                    faceAligned = fa.align(img, gray, dlibRect)
                    if dataset.mode == 'image':
                        if not os.path.exists(p):
                            os.makedirs(p)
                        print("img save _____________", img_save_path)
                        cv2.imwrite(img_save_path, faceAligned)


            else:
                for i in range(len(bboxes)):
                    faceimage = img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2]]
                    (fH, fW) = faceimage.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    if dataset.mode == 'image':
                        if not os.path.exists(p):
                            os.makedirs(p)
                        print("img save _____________", img_save_path)
                        cv2.imwrite(img_save_path, faceimage)

if __name__ == '__main__':
    parser = get_data_prepration_opt()
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    if opt.command == 'balance':
        balance_dataset()
    elif opt.command == 'create_faces':
        create_faces_dataset()
