from ultralytics import YOLO

model = YOLO("cfg/MACF-YOLO.yaml")

model.info()  # display model information

model.train(data="ultralytics/cfg/datasets/VisDrone.yaml",
            epochs=300,
            name="MACF-YOLO",
            imgsz=640,
            batch=2,
            device=0,
            resume=False,
            save_period=10,
            project="/mnt/e/Code/MACF-YOLO-main/result",
            pretrained=False,
            )  # train the model
