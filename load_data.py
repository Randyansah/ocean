import pathlib
import os
data_dir="./nasa_images"
data_dir=pathlib.Path(data_dir)
debris_path="./nasa_images/debris"
no_debris_path="./nasa_images/no_debris"
images_debris=os.listdir(debris_path)
images_no_debris=os.listdir(no_debris_path)