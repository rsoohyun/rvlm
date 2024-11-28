from .dataset import WaterBirdsDataset
from .transforms import transforms_preprcs


def load_dataset(data_dir, dataset_name, split):
    if dataset_name == "waterbird":
        dataset = WaterBirdsDataset(root=f"{data_dir}/{dataset_name}", split=split, transform=transforms_preprcs["waterbird"][split])    # 4795
    else:
        raise NotImplementedError(f'{dataset_name} is not supported.')
    
    return dataset
        