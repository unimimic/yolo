from ultralytics import YOLO

projectName = "test"
taskName = "1232"

task = 'train'
# task = 'export'

if task == 'train':
  # Load a model
  model = YOLO("./models/yolov8.yaml")
  # model = YOLO("yolov8n.pt") 

  # Train the model with MPS
  results = model.train(
    data=f"./projects/{projectName}/dataset/data.yaml",
    # epochs=100, imgsz=640,
    device="mps", # device=0
    name= taskName,
    project = f'./projects/{projectName}/output',
    cfg= f"./projects/{projectName}/args.yaml"
  )
elif task == 'export':
  # Load a model
  model = YOLO(f"./projects/{projectName}/output/{taskName}/weights/best.pt")  # load a custom trained model
  # Export the model saved_model
  # model.export(format="openvino")
  model.export(format="saved_model")