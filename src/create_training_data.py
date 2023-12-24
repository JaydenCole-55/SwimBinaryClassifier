import os
import cv2 as cv
from pathlib import Path
import numpy as n

TITLE = "Data2"

CWD = os.getcwd()
DATA_FLDR = Path(CWD, 'data', TITLE)

os.makedirs(DATA_FLDR, exist_ok=True)

video_file = Path(DATA_FLDR, TITLE + '.mp4')

print(video_file)
cap = cv.VideoCapture(str(video_file))

output_file = open(Path(DATA_FLDR, TITLE + ".csv"), 'w')
output_file.write("event,heat,start,end\n")

frame_num = 0
start_frame = 0
end_frame = 0

heat = 1

# Read in video
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)

    pressed_key = cv.waitKey(1)
    if pressed_key == ord('q'):
        print('Quit at frame: ' + str(frame_num))
        break
    elif pressed_key == ord('1'):
        start_frame = frame_num
    elif pressed_key == ord('0'):
        end_frame = frame_num

        output_string = "50fr," + str(heat) + "," + str(start_frame) + "," + str(end_frame) + "\n"
        output_file.write(output_string)

        start_frame = 0
        end_frame = 0
        heat = heat + 1

    frame_num = frame_num + 1

cap.release()
cv.destroyAllWindows()

output_file.close()