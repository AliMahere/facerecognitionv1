from opt import get_recogntion_opt
from face_detector import FaceDetector
from load_dataset import LoadImages
from face_embdings import FaceEmbder
from match import compare_faces
import os 
import cv2
import pickle


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def encode():
    source, saveEncoded, face_detection =  opt.dataset,  opt.encodings, opt.detection

    
    em = FaceEmbder(device = opt.device, model_name="openface")
    total = 0
    knownEmbeddings = []
    knownNames = []
    if face_detection:
        fd = FaceDetector(device = opt.device, conf_threshold = opt.confidence)
        for subdir, dirs, files in os.walk(source):
            print(subdir)
            dataset = LoadImages(subdir)
            for path, img, _ in dataset:
                name = path.split(os.path.sep)[-2]
                _ , bboxes =  fd.detect_frame(img)
                for i in range(len(bboxes)):
                    faceimage = img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2]]
                    (fH, fW) = faceimage.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    #cv2.imshow("face", faceimage)
                    knownNames.append(name)
                    embding= em.get_embdings(faceImg= faceimage)
                    knownEmbeddings.append(embding)
                    total += 1
    else :
        for subdir, dirs, files in os.walk(source):
            print(subdir)
            dataset = LoadImages(subdir)
            for path, img, _ in dataset:
                name = path.split(os.path.sep)[-2]
                knownNames.append(name)
                embding= em.get_embdings(faceImg= img)
                knownEmbeddings.append(embding)
                total += 1
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(saveEncoded, "wb")
    f.write(pickle.dumps(data))
    f.close()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def recognize():

    source, saveEncoded =  opt.input,  opt.embeddings
    fd = FaceDetector(device = opt.device, conf_threshold = opt.confidence)
    em = FaceEmbder(device = opt.device, model_name="openface")
    data = pickle.loads(open(saveEncoded, "rb").read())
    dataset = LoadImages(source)
    vid_path, vid_writer = None, None
    save_path = 'out.mp4'
    for  path , img, vid_cap in dataset:
        _ , bboxes =  fd.detect_frame(img)
        for i in range(len(bboxes)):
            faceimage = img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2]]
            (fH, fW) = faceimage.shape[:2]
            if fW < 20 or fH < 20:
                continue
            embding= em.get_embdings(faceImg= faceimage)
            matches = compare_faces(data["embeddings"], embding, opt.similarty, opt.similarty_threshold)
            #set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [x for (x, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for j in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][j]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
            plot_one_box(bboxes[i], img, color=[255,255,0], label=name, line_thickness=3)
    
        
        if dataset.mode == 'image':
            cv2.imwrite(save_path, img)
        else:  # 'video'
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img)
        cv2.imshow("fas", img)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    parser = get_recogntion_opt()
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    if opt.command == 'encode':
        encode()
    else:
        recognize()