# Prepare data for tensorflow
import os
import csv
from pathlib import Path
import cv2 as cv

title = "Data2"

cwd = os.getcwd()
image_path = Path(cwd,'data',title,'images')
race_path = Path(image_path,'race')
not_race_path = Path(image_path,'notRace') 

os.makedirs(image_path, exist_ok = True)
os.makedirs(race_path, exist_ok = True)
os.makedirs(not_race_path, exist_ok = True)

# Based on csv separate frames into each folder
# Determine the frame ranges for video
ranges = list()
with open(Path('data', title, title +'.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    csv_headers = next(reader)

    for row in reader:
        row = str(row)[:-2].split(',')
        ranges.append((int(row[2]), int(row[3])))

cap = cv.VideoCapture(str(Path('data', title, title + '.mp4')))

frame_num = 0
LAST_FRAME_NUM = 119247
frame_count = int(cv.VideoCapture.get(cap, int(cv.CAP_PROP_FRAME_COUNT)))
print("Total frames: " + str(frame_count))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_num > LAST_FRAME_NUM:
        break

    # Select every x frames
    if frame_num % 20:
        frame_num += 1
        continue

    # Save frame in either race or not race
    if any(lower <= frame_num <= upper for (lower, upper) in ranges):
        cv.imwrite(str(race_path) + '\\frame%d.jpg' % frame_num, frame)
    else:
        cv.imwrite(str(not_race_path) + '\\frame%d.jpg' % frame_num, frame)

    if not (frame_num % 1000):
        print("Current processed frame: " + str(frame_num))
    
    frame_num += 1

print("Processed " + str(frame_num) + " frames")