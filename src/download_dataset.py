from roboflow import Roboflow

rf = Roboflow(api_key="Z1Tq4adSnWFQu01FkxkT")

project = rf.workspace("ppe-ihvqu").project("ppe-8k2vo")
version = project.version(3)

dataset = version.download("yolov8")

print("Dataset saved at:", dataset.location)