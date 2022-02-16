
import argparse
def get_recogntion_opt():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(dest='command', required=True,
        help="chose command for recognize or encode type test.py /{command/} -h for help")

    parser.add_argument("-dm", "--detection-method", type=int, default=0,
        help="face detection model to use: either `0 ofr caffe` or `1 for tensorflow`")
    parser.add_argument("-dv", "--device", type=str, default='cpu',
        help="device used for dnn to use: either `cpu or `Gpu`")
    parser.add_argument("-em", "--encoding-method", type=int, default=0,
        help="face encoding model to use: either `0 ofr caffe` or `1 for tensorflow`") 
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
	    help="minimum probability to filter weak detections")

   
    recognize = subparser.add_parser('recognize')
    encode = subparser.add_parser('encode')

    encode.add_argument("-da", "--dataset",required=True,
	    help="path to input directory of faces + images")
    encode.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
    encode.add_argument("-de", "--detection", action="store_true",
      help="use it when you want to apply face detection on your dataset")

    recognize.add_argument("-i", "--input", required=True,
        help="input for recognize can be path for a set of imges/vedeos or poth of them , or path for single image or video ")
    recognize.add_argument("-e", "--embeddings", required=True,
	    help="path to serialized db of facial embeddings")
    recognize.add_argument("-ou", "--output", required= True,
        help="output file name ")
    recognize.add_argument("-si", "--similarty", default='cos',
	    help="similarty function chose cos or euclidean")
    recognize.add_argument("-st", "--similarty_threshold", type=float, default=0.8, 
        help="similarity-threshold for compairing faces to detect unknown class ")
    recognize.add_argument('-aline', action='store_true')

    return parser


def get_data_prepration_opt():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command', required=True,
        help="chose command for recognize or encode type test.py /{command/} -h for help")   
    balance = subparser.add_parser('balance')
    create_faces = subparser.add_parser('create_faces')
    
    parser.add_argument("-da", "--dataset",required=True,
    help="path to input directory of faces + images")


    balance.add_argument("-sa", "--spamples",type=int ,required=True,
        help="requred sample number  ")



    create_faces.add_argument("-dm", "--detection-method", type=int, default=0,
        help="face detection model to use: either `0 ofr caffe` or `1 for tensorflow`")
    create_faces.add_argument("-dv", "--device", type=str, default='cpu',
        help="device used for dnn to use: either `cpu or `Gpu`")

    create_faces.add_argument("-c", "--confidence", type=float, default=0.5,
	    help="minimum probability to filter weak detections")
    create_faces.add_argument("-pr", "--padding_ratio", type=float, default=0,
	    help="minimum probability to filter weak detections")

    create_faces.add_argument("-sp", "--save_path",required=True,
        help="path to output directory of faces + images")
    create_faces.add_argument('-aline', action='store_true')




    return parser


# def print_options( opt):
#     message = ''
#     message += '----------------- Options ---------------\n'
#     for k, v in sorted(vars(opt).items()):
#         comment = ''
#         default = parser.get_default(k)
#         if v != default:
#             comment = '\t[default: %s]' % str(default)
#         message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
#     message += '----------------- End -------------------'
#     print(message)