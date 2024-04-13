from ultralytics import YOLO
import pandas as pd

if __name__ == "__main__":
    # Load an official or custom model
    model = YOLO('yolov8n.pt')  # Load an official Detect model

    # Perform tracking with the model
    results = model.track(source="https://www.youtube.com/watch?v=zBBVnq20HFU",
                          show=True, save=False)  # , stream_buffer=True)  # Tracking with default tracker

    numbers_of_objects = []
    all_classes = []
    objects_ids = []

    names = model.names
    for result in results:
        objects_ids.append([int(x) for x in list(result.boxes.id.numpy())])
        numbers_of_objects.append(len(result.boxes.cls))
        all_classes.append([names[int(cls)] for cls in result.boxes.cls])

    csv_file_header = {'numbers_of_objects': numbers_of_objects,
                       'all_classes': all_classes,
                       'objects_ids': objects_ids}
    dataframe = pd.DataFrame(csv_file_header)

    dataframe.to_csv('report.csv')
