import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt

import cv2
from time import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_video_path',
                    help='video to be processed')
parser.add_argument('-o', dest='vid_record_path',
                    help='recorded video path')
args = parser.parse_args()

INPUT_VIDEO_PATH =  args.input_video_path
VID_RECORD_PATH = args.vid_record_path

print ('input_video_path     =', INPUT_VIDEO_PATH)
print ('recorded_video_path     =', VID_RECORD_PATH)

def VideoSrcInit():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    #cap = skvideo.io.vread(INPUT_VIDEO_PATH)
    flag, image = cap.read()
    if flag:
        print("Valid Video Path. Lets move to detection!")
    else:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")
    return cap

def VideoRecInit(WIDTH,HEIGHT):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(VID_RECORD_PATH, fourcc, 5.0, (WIDTH,HEIGHT))
    return videowriter


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)


cap = VideoSrcInit()
flag, image = cap.read()
(ht,wd,_) = image.shape
videowriter = VideoRecInit(wd,ht)

frame_no = 1

while(True):
	start_time = time()
	cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
	flag, image = cap.read()
	if flag == False:
		break


	image_batch = data_to_input(image)

	# Compute prediction with the CNN
	outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
	scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

	detections = extract_detections(cfg, scmap, locref, pairwise_diff)
	unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
	person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

	img = np.copy(image)

	visim_multi = img.copy()

	img = draw_multi.draw(visim_multi, dataset, person_conf_multi)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	frame_no += 3

	#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
	#cv2.imshow('Image',img)
	videowriter.write(img)

	print("FPS: ", 1.0 / (time() - start_time))
	# Continue until the user presses ESC key

videowriter.release()
cap.release()


