import cv2
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from assembledModel.detector.Detector import Detector


def changeColor():
    colorOptions = [0, 128, 255]
    b = 0
    g = 0
    r = 0

    while (b, g, r) == (0, 0, 0) or (b, g, r) == (255, 255, 255):
        b = random.choice(colorOptions)
        g = random.choice(colorOptions)
        r = random.choice(colorOptions)

    colorMap = (b, g, r)

    return colorMap


def DepthMap(type, img):
    width = img.shape[1]
    height = img.shape[0]

    path_model = "midas/"
    if type == "fast":
        blob = cv2.dnn.blobFromImage(img, 1 / 255., (256, 256), (123.675, 116.28, 103.53), True, False)
        model = "model-small.onnx"
    elif type == "accurate":
        blob = cv2.dnn.blobFromImage(img, 1 / 255., (384, 384), (123.675, 116.28, 103.53), True, False)
        model = "model-f6b98070.onnx"
    else:
        raise IOError(Exception, "Depth module type not correct. Choose from <fast> or <accurate>.")

    midas = cv2.dnn.readNet(path_model + model)

    midas.setInput(blob)
    output = midas.forward()

    output = output[0, :, :]

    output = cv2.resize(output, (width, height))
    depthMap = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return depthMap

def displayDepthPointOnMask(depthMap, depthPoint):
    x = depthPoint[1]
    y = depthPoint[0]
    black = (0, 0, 0)

    depthMap[y][x] = black
    depthMap[y][x-1] = black
    depthMap[y][x+1] = black
    depthMap[y-1][x] = black
    depthMap[y+1][x] = black
    depthMap[y][x-2] = black
    depthMap[y][x+2] = black
    depthMap[y-2][x] = black
    depthMap[y+2][x] = black
    depthMap[y][x-3] = black
    depthMap[y][x+3] = black
    depthMap[y-3][x] = black
    depthMap[y+3][x] = black
    depthMap[y][x-4] = black
    depthMap[y][x+4] = black
    depthMap[y-4][x] = black
    depthMap[y+4][x] = black

    return depthMap

# init Detectron2
detector = Detector()

# read and display original image
img = cv2.imread("images/test6.jpeg")
cv2.imshow("Normal Image", img)
width = img.shape[1]
height = img.shape[0]

# compute MiDaS output
start_time = time.time()
depthMap = DepthMap("fast", img)
end_time = time.time() - start_time
print(str(round(end_time, 2)) + " seconds for MiDaS")

# save original depth map values
depthMapCopy = depthMap
cv2.imshow("MiDaS", depthMapCopy)

# normalize depth map to RGB
depthMap = cv2.cvtColor(depthMap, cv2.COLOR_GRAY2RGB)

start_time = time.time()
frame, data = detector.predict(img)
end_time = time.time() - start_time
print(str(round(end_time, 2)) + " seconds for Detectron2")
cv2.imshow("Detectron2", frame)

# get Detectron2 output
iBoxes = data["instances"].pred_boxes
iMasks = data["instances"].pred_masks
iClasses = data["instances"].pred_classes.tolist()
iScores = data["instances"].scores.tolist()

numOfInstances = len(iClasses)

# save original color map values
boxColor = changeColor()

for instIndex in range(len(iClasses)):
    print("Computing for instance with class: " + str(iClasses[instIndex]))

    box = iBoxes[instIndex].tensor.tolist()[0]
    print(box)
    topLeft = (int(box[0]), int(box[1]))
    bottomRight = (int(box[2]), int(box[3]))

    xMid = int((topLeft[0] + bottomRight[0]) / 2)

    boxThickness = 2

    print("topLeft: " + str(topLeft) + ", bottomRight: " + str(bottomRight))
    xMargin = topLeft[0]
    yMargin = topLeft[1]

    print("x, y Margin: " + str(xMargin) + " " + str(yMargin))

    mask = iMasks[instIndex].tolist()

    # find the instance's center of mass for MiDaS
    for yUp in range(yMargin, bottomRight[1]):
        if mask[yUp][xMid]:
            break

    for yDown in range(yUp, bottomRight[1]):
        if not mask[yDown][xMid]:
            break

    yMid = int((yUp + yDown)/2)

    depthPoint = (yMid, xMid)
    depthValue = round(depthMapCopy[depthPoint], 4)

    print("Found depth point at: " + str(depthPoint) + ". Depth value: " + str(depthValue))

    instanceColor = (boxColor[0] / 2, boxColor[1] / 2, boxColor[2] / 2)

    for x in range(xMargin, bottomRight[0]):
        for y in range(yMargin, bottomRight[1]):
            if mask[y][x]:
                depthMap[y][x] = instanceColor

    # represent Depth Point on Detectron2 mask
    depthMap = displayDepthPointOnMask(depthMap, depthPoint)

    print("instance colored")

    depthMap = cv2.rectangle(depthMap, topLeft, bottomRight, boxColor, boxThickness)
    depthMap = cv2.putText(depthMap, str(iClasses[instIndex]), (topLeft[0] + 2, topLeft[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, boxColor, 2, cv2.LINE_AA)
    depthMap = cv2.putText(depthMap, str(depthValue), (topLeft[0], bottomRight[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, boxColor, 2, cv2.LINE_AA)

    # change color scheme for different instances
    boxColor = changeColor()

# verify
cv2.imshow("new MiDaS", depthMap)
cv2.waitKey(0)

# capture = cv2.VideoCapture(1)
#
# start_time = time.time()
#
# if capture.isOpened():
#     print("Capture open")
# else:
#     raise IOError("Capture not working")
#
# end_time = 0
# fps = 0
#
# while True:
#     _, frame = capture.read()
#
#     h = int(frame.shape[0] * 0.4)
#     w = int(frame.shape[1] * 0.4)
#     dim = (w, h)
#     frame = cv2.resize(frame, dim)
#
#     newFrame = detector.predict(frame)
#     end_time = time.time()
#     fps = round(1 / (end_time - start_time), 2)
#     start_time = end_time
#     frameWithFPS = cv2.putText(frame, str(fps) + " FPS", (17, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
#                                    cv2.LINE_AA)
#
#     print(newFrame.shape)
#     print(type(frameWithFPS))
#
#     cv2.imshow("Camera", frameWithFPS)
#     cv2.imshow("IS", newFrame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# capture.release()
# cv2.destroyAllWindows()
