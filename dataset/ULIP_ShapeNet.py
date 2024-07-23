import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import os


DATASET_ADDRESS = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets/"


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def create_image_grid(images):
    # 创建一个图像网格
    grid_size = int(np.ceil(np.sqrt(len(images))))
    grid_img = Image.new('RGB', (images[0].width * grid_size, images[0].height * grid_size))
    for i, img in enumerate(images):
        grid_img.paste(img, ((i % grid_size) * img.width, (i // grid_size) * img.height))
    return grid_img


class ULIP_ShapeNet(Dataset):
    def __init__(self, dataset_path=DATASET_ADDRESS, keyword=None, grounding_type=None, sample_angle_range=60, image_size=512, random_flip=False, prob_use_caption=1):
        self.dataset_path = dataset_path

        self.path_caption_data = os.path.join(dataset_path, 'captions')
        self.path_data_pc = os.path.join(dataset_path, "shapenet_pc")
        self.path_data_rgb = os.path.join(dataset_path, "only_rgb_depth_images")

        self.all_angles = np.arange(0, 360, 12)
        self.sample_angle_range = sample_angle_range

        self.keyword = keyword
        if not self.keyword:
            self.pointcloud_filename_list = sorted(os.listdir(self.path_data_pc))
        else:
            json_file = os.path.join(dataset_path, "filter_by_keyword", self.keyword + ".json")
            self.pointcloud_filename_list = load_json(json_file)

        self.image_size = image_size
        self.random_flip = random_flip
        self.prob_use_caption = prob_use_caption

        self.pil_to_tensor = transforms.PILToTensor()

        self.grounding_type = grounding_type
        if grounding_type == None:
            self.source_name = "source"
            self.target_name = "target"
        elif grounding_type == "canny":
            self.source_name = "canny_edge"
            self.target_name = "image"
        elif grounding_type == "depth":
            self.source_name = "depth"
            self.target_name = "image"
        elif grounding_type == "hed":
            self.source_name = "hed_edge"
            self.target_name = "image"
        else:
            raise Exception("Not supported grounding type.")


    def __len__(self):
        return len(self.pointcloud_filename_list)
    

    def total_images(self):
        return len(self)
    

    def process_index(self, index=None, show_images=False):
        if index is None:
            index = np.random.randint(len(self))
            
        name = self.pointcloud_filename_list[index]

        if name.endswith(".npy"):
            name = name[:-4]
        
        pc_np = np.load(os.path.join(self.path_data_pc, name + ".npy"))
        # print("pc_np.shape: ")
        # print(pc_np.shape)
        
        data = {'pointcloud_np': pc_np}
        
        captions_data = load_json(os.path.join(self.path_caption_data, name + ".json"))
        
        caption_missing = 0
        
        RGB_imgs = []
        for i, angle in enumerate(self.all_angles):
            img_name = name + f"_r_{angle:03d}.png"
            
            img_path = os.path.join(self.path_data_rgb, img_name)
            
            if img_name in captions_data:
                captions_rgb = captions_data[img_name]
            else:
                captions_rgb = [""] * 10  # 如果找不到对应的描述，使用空描述
                caption_missing += 1
                
            img_a = Image.open(img_path).convert("RGB")
            
            if show_images:
                RGB_imgs.append(img_a)
            
            data[f'angle_{i+1}'] = {
                'angle': angle,
                'image': img_a,
                'captions': captions_rgb,
            }
        
        if caption_missing > 0:
            print("!!!" + str(caption_missing) + " images & captions are missing!!!")
        
        if show_images:
            RGB_imgs_show = create_image_grid(RGB_imgs)
            
            return data, RGB_imgs_show
        else:
            return data
        

    def choose_random_angles(self, data, angle_range=90):
        # 获取所有角度信息
        angles = [data[key]['angle'] for key in data if key.startswith('angle_')]

        # 随机选择角度A及其索引
        index_A = np.random.randint(len(angles))
        angle_A = angles[index_A]
        
        # 确定B的范围
        min_angle_B = (angle_A - angle_range) % 360
        max_angle_B = (angle_A + angle_range) % 360
        
        # 生成B的有效索引范围
        if min_angle_B < max_angle_B:
            valid_indices = [i for i, angle in enumerate(angles) if min_angle_B <= angle <= max_angle_B]
        else:  # 处理角度循环
            valid_indices = [i for i, angle in enumerate(angles) if angle >= min_angle_B or angle <= max_angle_B]
            
        if not valid_indices:
            return None, None

        # 从有效索引中随机选择角度B及其索引
        index_B = np.random.choice(valid_indices)
        angle_B = angles[index_B]
        
        return index_A + 1, index_B + 1


    def __getitem__(self, idx):


        data = self.process_index(idx)

        RETRY_MAX = 10
        retry = 0

        while retry < RETRY_MAX:
            index_source, index_target = self.choose_random_angles(data, self.sample_angle_range)

            if index_source and index_target:
                break
            
            retry += 1


        if not index_source or not index_target:
            # not valid data pair found, return empty template
            return {
                'id': idx,
                self.target_name: torch.zeros(3, self.image_size, self.image_size),
                self.source_name: torch.zeros(3, self.image_size, self.image_size),
                'mask': torch.tensor(1.0),
                'caption': ""
            }


        # source
        angle_source = data[f"angle_{index_source}"]["angle"]
        source = data[f"angle_{index_source}"]["image"]

        # target
        angle_target = data[f"angle_{index_target}"]["angle"]
        target = data[f"angle_{index_target}"]["image"]

        # rotation between the source and target
        rotation = angle_target - angle_source
        if rotation < -180:
            rotation += 360
        elif rotation > 180:
            rotation -= 360

        # prompt guidance
        caption_idx = np.random.randint(len(data[f"angle_{index_target}"]["captions"]))
        caption = data[f"angle_{index_target}"]["captions"][caption_idx]
        caption = f"Image of a 3D rendering object, {caption}, with {rotation} degree rotating based on the reference image."


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
            self.target_name: target,
            self.source_name: source,
            'mask': torch.tensor(1.0),
            'caption': caption if random.uniform(0, 1) < self.prob_use_caption else ""
        }

        return out


if __name__ == "__main__":
    # keyword = "chair"
    # keyword = "table"
    keyword = "plane"
    # keyword = None

    dataset = ULIP_ShapeNet(keyword=keyword)
    print(len(dataset))
    print(dataset.total_images())

    item = dataset[34]
    print(item['caption'])
    print(item[dataset.target_name].shape)
    print(item[dataset.source_name].shape)
