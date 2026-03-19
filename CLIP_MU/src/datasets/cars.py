import os
import torch
from datasets import load_dataset
from PIL import Image

class PytorchStanfordCarsHF(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        pil_image = item["image"].convert("RGB")
        target = item["label"]
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, target

class Cars:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=32, num_workers=16):
        print("Note: Loading Stanford Cars from reliable HuggingFace mirror (uses cache after first download)...")
        self.hf_train = load_dataset('tanganke/stanford_cars', split='train')
        self.hf_test = load_dataset('tanganke/stanford_cars', split='test')

        self.classnames = [name.replace('_', ' ') for name in self.hf_train.features['label'].names]
        
        self.train_dataset = PytorchStanfordCarsHF(self.hf_train, preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PytorchStanfordCarsHF(self.hf_test, preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
