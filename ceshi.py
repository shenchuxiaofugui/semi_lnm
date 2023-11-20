from monai.data.dataset import Dataset
import json
from monai.transforms import LoadImaged

with open("/homes/dxli/Code/MRVesselSegmentation/brain_MRA_train.json") as file:
    data_list = json.load(file)

print(data_list[:2])
ds = Dataset(data_list[:2], transform=LoadImaged("image"))
for i in ds:
    print(i["image"].shape)


    
