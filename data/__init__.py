from .dataset import WaterBirdsDataset, CelebA
from .transforms import transforms_preprcs
from .templates import imagenet_templates


def load_dataset(data_dir, dataset_name, split, transform=None, prompt_id=0):
    if transform is None:
        transform = transforms_preprcs[dataset_name][split]
    
    if dataset_name == "waterbird":
        dataset = WaterBirdsDataset(root=f"{data_dir}/{dataset_name}", split=split, transform=transform, prompt_id=prompt_id)    # 4795
    elif dataset_name == "celeba":
        dataset = CelebA(root=f"{data_dir}/{dataset_name}", split=split, transform=transform, prompt_id=prompt_id)
    else:
        raise NotImplementedError(f'{dataset_name} is not supported.')
    
    return dataset
