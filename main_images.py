import cv2
import time
import os
import argparse
import re

from rknnpool import rknnPoolExecutor
# Image processing function, needs to be modified as required in actual applications
from func import myFunc



def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


# Define a function to extract the numeric part of the filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


current_directory = os.getcwd()
print("Current Directory:", current_directory)

base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir:", base_dir)

parser = argparse.ArgumentParser(description='Process some integers.')
# # basic params
# parser.add_argument('--model_path', type=str, default='./model/yolov8n.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--model_path', type=str, default='rknnModel/bucket_box_ball_epochs_100.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
parser.add_argument('--images_folder', type=str, default='images', help='images folder path for inference')
    

args = parser.parse_args()
print(vars(args))


# video_path = os.path.join(base_dir, args.video_path)
# cap = cv2.VideoCapture(video_path)

images_folder = os.path.join(base_dir, args.images_folder)


#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# modelPath = "./rknnModel/yolov8s.rknn"
model_path = os.path.join(base_dir, args.model_path)

# Number of threads, increasing this can improve the frame rate
TPEs = 4
# Initialize the rknn pool
pool = rknnPoolExecutor(
    rknnModel=model_path,
    TPEs=TPEs,
    func=myFunc)



# Get the list of files
file_list = os.listdir(images_folder)

# Sort the list using the custom key
file_list = sorted(file_list, key=extract_number)
#print(file_list)


img_list = []
for path in file_list:
    if img_check(path):
        img_list.append(path)


i=0

# Initialize the frames needed for asynchronous processing
if (len(img_list) > 0):
    for k in range(TPEs + 1):
        #print(i)
        img_name = img_list[i]
        img_path = os.path.join(images_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        pool.put(frame)
        i+=1


frames, loopTime, initTime = 0, time.time(), time.time()
while True:
    #print(i)
    img_name = img_list[i]
    img_path = os.path.join(images_folder, img_name)
    frames += 1
    frame = cv2.imread(img_path)
    if frame is None:
        continue
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
    
    i+=1
    if i==len(img_list):
        i=0

print("Overall average frame rate\t", frames / (time.time() - initTime))
# Release cap and rknn thread pool

cv2.destroyAllWindows()
pool.release()
