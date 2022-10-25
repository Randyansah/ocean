from PIL import Image
from load_data import images_debris,images_no_debris,data_dir

def coun():
    for num_debris in images_debris:
        print(num_debris)
    for num_no_debris in images_no_debris:
        print(num_no_debris)

    debris=list(data_dir.glob('debris/*'))
    Image.open(str(debris[0]))