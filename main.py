from ultralytics import YOLO
import torch
torch.cuda.set_device(0)

projectName = "test"
taskName = "24-10-06"

task = 'train'
# task = 'export'

if __name__ == "__main__":
  if task == 'train':
    # Load a model
    model = YOLO("./models/yolo11n.yaml").load("./models/yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO("yolov8n.pt") 

    # Train the model with MPS
    results = model.train(
      data=f"./projects/{projectName}/dataset/data.yaml",
      epochs=100000,
      patience=300,
      batch=8,
      device=0, # device=0 mps
      name= taskName,
      project = f'./projects/{projectName}/output',
      cfg= f"./projects/{projectName}/args.yaml"
    )
  elif task == 'export':
    # Load a model
    model = YOLO(f"./projects/{projectName}/output/{taskName}/weights/best.pt")  # load a custom trained model
    # Export the model saved_model
    model.export(format="openvino")
    # model.export(format="saved_model")