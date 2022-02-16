import Augmentor
import os
from face_detector import FaceDetector
from load_dataset import LoadImages
from pathlib import Path
from random import randint
import cv2


from opt import get_data_prepration_opt

def balance_dataset():
    source, target_num_of_spamples, rotate_degree, save_path =  opt.dataset, opt.spamples, opt.rotate_degree, opt.save_path
    for subdir, dirs, files in os.walk(source):
        if len(files) == 0:
            continue
        p = str(Path(subdir).absolute())
        save_dir =os.path.join(save_path ,p.split(os.path.sep)[-1] )
        if not os.path.exists(save_dir):
            print("false")
            os.makedirs(save_dir)
        print("___________len__________",len(files))

        p = Augmentor.Pipeline(subdir)

        p.rotate(probability=0.2, max_left_rotation=rotate_degree, max_right_rotation=rotate_degree)
        p.zoom(probability=0.2, min_factor=1.1, max_factor=1.3)
        p.flip_left_right(probability=0.2)
        p.crop_random(probability=0.4 , percentage_area = 0.7)
        p.sample(target_num_of_spamples - len(files)+2)

def create_faces_dataset():
    source, save_path =  opt.dataset, opt.save_path
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
