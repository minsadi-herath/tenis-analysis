#import all the libraries
from ultralytics import YOLO

#Load the YOLOv12 model
model = YOLO("yolo12n.pt")

#predictions
#results = model.predict(
    #source="input_video/input_video1.mp4",
    #save=True,
    #project="runs",      # folder inside project
    #name="tennis_output" # subfolder name
#)
results = model.track(
    source="input_video/input_video1.mp4",
    save=True,
    project="runs",      # folder inside project
    name="tennis_output", # subfolder name
    persist = True
)


