import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF

DATASET_ADDRESS = "/home/haiyang/1_Repo/ControlNet/training/fill50k/"


class ToyDataset(Dataset):
    def __init__(self, dataset_address=DATASET_ADDRESS, image_size=512, random_flip=False, prob_use_caption=1):
        self.dataset_address = dataset_address
        self.data = []
        self.image_size = image_size
        self.random_flip = random_flip
        self.prob_use_caption = prob_use_caption

        with open(self.dataset_address + 'prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.pil_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.data)
    
    def total_images(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(self.dataset_address + source_filename)
        target = cv2.imread(self.dataset_address + target_filename)

        # Convert images from BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Convert images to PIL format
        source = Image.fromarray(source)
        target = Image.fromarray(target)

        # Apply center crop, resize, and random flip
        assert  source.size == target.size

        crop_size = min(source.size)
        source = TF.center_crop(source, crop_size)
        source = source.resize((self.image_size, self.image_size))

        target = TF.center_crop(target, crop_size)
        target = target.resize((self.image_size, self.image_size))

        if self.random_flip and random.random() < 0.5:
            source = ImageOps.mirror(source)
            target = ImageOps.mirror(target)

        # Normalize images
        source = (self.pil_to_tensor(source).float() / 255.0 - 0.5) / 0.5
        target = (self.pil_to_tensor(target).float() / 255.0 - 0.5) / 0.5

        # Prepare output
        out = {
            'id': idx,
            'image': target,
            'canny_edge': source,
            'mask': torch.tensor(1.0),
            'caption': prompt if random.uniform(0, 1) < self.prob_use_caption else ""
        }

        return out

if __name__ == "__main__":
    dataset = ToyDataset()
    print(len(dataset))

    item = dataset[1234]
    print(item['caption'])
    print(item['image'].shape)
    print(item['canny_edge'].shape)