import cv2
import time
import os
import argparse

from rknnpool import rknnPoolExecutor
# Image processing function, needs to be modified as required in actual applications
from func import myFunc

current_directory = os.getcwd()
print("Current Directory:", current_directory)

base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir:", base_dir)

parser = argparse.ArgumentParser(description='Process some integers.')
# # basic params
# parser.add_argument('--model_path', type=str, default='./model/yolov8n.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--model_path', type=str, default='rknnModel/bird_1_3.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
parser.add_argument('--video_path', type=str, default='video/004.mp4', help='video path for inference')
    

args = parser.parse_args()
print(vars(args))


video_path = os.path.join(base_dir, args.video_path)

cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# modelPath = "./rknnModel/yolov8s.rknn"
model_path = os.path.join(base_dir, args.model_path)

# Number of threads, increasing this can improve the frame rate
TPEs = 6
# Initialize the rknn pool
pool = rknnPoolExecutor(
    rknnModel=model_path,
    TPEs=TPEs,
    func=myFunc)

# Initialize the frames needed for asynchronous processing
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    #print(frame.shape)
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    #print(frame.shape)
    cv2.imshow('yolov8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("Average frame rate for 30 frames:\t", 30 / (time.time() - loopTime), "frames")
        loopTime = time.time()

print("Overall average frame rate\t", frames / (time.time() - initTime))
# Release cap and rknn thread pool
cap.release()
cv2.destroyAllWindows()
pool.release()
