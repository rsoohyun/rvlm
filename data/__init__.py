from .dataset import WaterBirdsDataset
from .transforms import transforms_preprcs


def load_dataset(data_dir, dataset_name, split, transform=None):
    if transform is None:
        transform = transforms_preprcs["waterbird"][split]
    
    if dataset_name == "waterbird":
        dataset = WaterBirdsDataset(root=f"{data_dir}/{dataset_name}", split=split, transform=transform)    # 4795
    else:
        raise NotImplementedError(f'{dataset_name} is not supported.')
    
    return dataset
        