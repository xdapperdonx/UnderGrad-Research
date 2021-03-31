import logging
import argparse
import xml.etree.cElementTree as ET
import numpy as np
import sys
import os
from utils import *

logging.basicConfig(
    filename='logfile',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Executing program... \n------------------------")

#terminal input: python3 yoloface.py --folder data/ --output-dir outputs/
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--folder', type=str, default='',
                    help='path to folder')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def predictedAnnotations():

    faces_detected = 0
    input_files = []
    output_file = ''

    if args.folder:

        #Error if no folder if found.
        if not os.path.isdir(args.folder):
            print("[!] ==> Input directory {} doesn't exist".format(args.folder))
            sys.exit(1)
        
        for filename in os.listdir(args.folder):
            cap = cv2.VideoCapture(os.path.join(args.folder, filename))
            output_file = filename[:-4].rsplit('/')[-1] + '_yoloface.jpg'

            while True:
                has_frame, frame = cap.read()
 
                if not has_frame:
                    cv2.waitKey(1000)
                    break

                # Create a 4D blob from a frame.
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                            [0, 0, 0], 1, crop=False)

                # Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers
                outs = net.forward(get_outputs_names(net))

                # Remove the bounding boxes with low confidence
                faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                logging.info('[i] ==> # detected faces: {}'.format(len(faces)))
                logging.info(f"{filename}: {faces}\n------------------------")
                faces_detected += len(faces)

                # initialize the set of information we'll displaying on the frame
                info = [
                    ('number of faces detected', '{}'.format(len(faces)))
                ]

                for (i, (txt, val)) in enumerate(info):
                    text = '{}: {}'.format(txt, val)
                    cv2.putText(frame, text, (10, (i * 20) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

                # Save the output video to file
                if args.folder:
                    cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))

                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    print('[i] ==> Interrupted by user!')
                    break

            cap.release()
            cv2.destroyAllWindows()
    return faces

def realAnnotation():
    path_xml = '/home/xdapperdonx/Desktop/Code/Undergraduate/YoloAccuracyV3/annotations'
    xmls = []
    extract_annotations = []
    counter = 0

    #stores xmls file in array
    for file_xml in os.listdir(path_xml):
        if file_xml.endswith('.xml'):
            xmls.append(file_xml)

    for data in xmls:
        tree = ET.ElementTree(file=os.path.join(path_xml,data))
        root = tree.getroot()

        #extracts the bounding box
        for elem in root.findall('./object/bndbox/'):
                extract_annotations.append(int(elem.text))

    #splices list and stores a list of list
    return extract_annotations

def calAccuracy(real, predicted_annotations):

    real_annotations = np.array([real[x:x+4] for x in range(0, len(real), 4)])
    predicted_annotations = np.array(predicted_annotations)
    real_index = None
    predicted_index = None
    compare = np.linalg.norm(real_annotations[0]-predicted_annotations[0])
    paired_annotations = []

    while(len(real_annotations) != 0):

        #finds most similar values for predicted and real annotations
        for i in real_annotations:
            for j in predicted_annotations:
                if np.linalg.norm(i-j) < compare:
                    compare = np.linalg.norm(i-j)
                    real_index = i
                    predicted_index = j
                elif(len(real_annotations) == 1):
                    real_index = i
                    predicted_index = j
                    paired_annotations.append(predicted_index.tolist())
                    paired_annotations.append(real_index.tolist())
                    break

        #appends to paired_annotations closest [[predicted], [actual]]
        paired_annotations.append(predicted_index.tolist())
        paired_annotations.append(real_index.tolist())

        #deletes where annotations in original array
        real_annotations = np.delete(real_annotations, np.where(real_annotations == real_index))

        #reformat as a 2d array
        real_annotations = np.array([real_annotations[x:x+4] for x in range(0, len(real_annotations), 4)])

        if(real_annotations.size != 0):
            compare = np.linalg.norm(real_annotations[0]-predicted_annotations[0])

    #split array into pred and real
    pred = paired_annotations[::2]
    real = paired_annotations[1::2]

    pred_sum = []
    real_sum = []

    for pred_elem in pred:
        #sums xmin ymin and xmax ymax
        #pred_sum.append([i+j for i,j in zip(pred_elem[::2], pred_elem[1::2])])

        #sum just xmin ymin
        firsttwo = [i+j for i,j in zip(pred_elem[:2:2], pred_elem[1:2:2])]
        pred_sum.append(firsttwo[0])
  
    for real_elem in real:
        #sums xmin ymin and xmax ymax
        #real_sum.append([i+j for i,j in zip(real_elem[::2], real_elem[1::2])])

        #sum just xmin ymin
        firsttwo = [i+j for i,j in zip(real_elem[:2:2], real_elem[1:2:2])]
        real_sum.append(firsttwo[0])
    
    pred_sum = np.array(pred_sum)
    real_sum = np.array(real_sum)
    for i in range(0, len(real_sum)):
        dist = np.linalg.norm(pred_sum[i] - real_sum[i])
        print(dist)

    
if __name__ == '__main__':
    calAccuracy(realAnnotation(), predictedAnnotations())