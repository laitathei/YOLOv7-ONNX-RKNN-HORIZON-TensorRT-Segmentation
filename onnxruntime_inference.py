import cv2, os
import numpy as np
import onnxruntime
from util import *

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 480
result_path = "./result"
image_path = "./dataset/bus.jpg"
model_name = 'yolov7-seg'
model_path = "./model"
ONNX_MODEL = f"{model_path}/{model_name}-{input_height}-{input_width}.onnx"
video_path = "test.mp4"
video_inference = False
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']

sess = onnxruntime.InferenceSession(ONNX_MODEL)
isExist = os.path.exists(result_path)
if not isExist:
    os.makedirs(result_path)

if video_inference == True:
    cap = cv2.VideoCapture(video_path)
    while(True):
        ret, image_3c = cap.read()
        if not ret:
            break
        print('--> Running model for video inference')
        image_4c, image_3c = preprocess(image_3c, input_height, input_width)
        outputs = sess.run(['output', '528', '638', '748', '513'],{"images": image_4c.astype(np.float32)}) # (1, 3, input height, input width)
        colorlist = gen_color(len(CLASSES))
        outputs = [outputs[0], outputs[4]]
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, masks, shape = results
        if isinstance(masks, np.ndarray):
            mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            cv2.imshow("mask_img", mask_img)
            cv2.imshow("vis_img", vis_img)
        else:
            print("No segmentation result")
        cv2.waitKey(10)
else:
    image_3c = cv2.imread(image_path)
    image_4c, image_3c = preprocess(image_3c, input_height, input_width)
    outputs = sess.run(['output', '528', '638', '748', '513'],{"images": image_4c.astype(np.float32)}) # (1, 3, input height, input width)
    colorlist = gen_color(len(CLASSES))
    outputs = [outputs[0], outputs[4]]
    results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres) ##[box,mask,shape]
    results = results[0]              ## batch=1
    boxes, masks, shape = results
    if isinstance(masks, np.ndarray):
        mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
        print('--> Save inference result')
    else:
        print("No segmentation result")


print("ONNX inference finish")
cv2.destroyAllWindows()


