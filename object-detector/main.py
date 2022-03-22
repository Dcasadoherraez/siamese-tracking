import cv2
import torch

from yolort.utils import Visualizer, get_image_from_url, read_image_to_tensor
from yolort.v5.utils.downloads import safe_download
from yolort.models import yolov5n6

import os

# Vis
import io
import contextlib
from torchvision.ops import box_convert
from yolort.data import COCODetectionDataModule
from yolort.models.transform import YOLOTransform
from yolort.utils.image_utils import (
    color_list,
    plot_one_box,
    cv2_imshow,
    load_names,
    parse_single_image,
    parse_images,
)

#  Get COCO label names and COLORS list
LABELS = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
)

COLORS = color_list()   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read image
img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/bus.jpg"
# img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/zidane.jpg"



def predict_frame(img_raw):
    img = read_image_to_tensor(img_raw)
    img = img.to(device)

    images = [img]
    # Define and initialize model
    score_thresh = 0.55
    # stride = 64
    # img_h, img_w = 640, 640

    model = yolov5n6(pretrained=True, score_thresh=score_thresh)
    model = model.eval()
    model = model.to(device)

    # Perform inference on an image tensor
    model_out = model(images)

    #########################################################
    img_raw = cv2.cvtColor(parse_single_image(img), cv2.COLOR_RGB2BGR)  # For visualization

    for box, label in zip(model_out[0]['boxes'].tolist(), model_out[0]['labels'].tolist()):
        img_raw = plot_one_box(box, img_raw, color=COLORS[label % len(COLORS)], label=LABELS[label])

    return img_raw

cap = cv2.VideoCapture("videoplayback.mp4")
if not cap.isOpened():
    print("This does not work")
    exit(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:

        prediction = predict_frame(frame)

        cv2.imshow("Test", prediction)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
