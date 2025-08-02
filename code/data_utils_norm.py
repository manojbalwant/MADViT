import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import pdb
import torch.utils.data as data
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from gaussian_blur import GaussianBlur
import random
from torchvision import transforms, datasets
from tqdm import tqdm

def compute_mean_and_std(train_folder, resize_height, resize_width, device, batch_size=64, num_workers=2):
    # Define the image transformation: resize and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
    ])
    
    # Create the dataset and data loader.
    dataset = datasets.ImageFolder(train_folder, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Determine the number of channels (e.g., 3 for RGB)
    channels = dataset[0][0].shape[0]
    
    # Initialize variables to accumulate the sums and squared sums.
    mean = torch.zeros(channels).to(device)  # Use GPU for mean
    std = torch.zeros(channels).to(device)   # Use GPU for std
    total_pixels = 0  # Total number of pixels per channel across all images.
    
    # Ensure no gradients are tracked.
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing mean and std"):
            images = images.to(device)  # Move images to GPU
            
            batch_size, c, h, w = images.shape
            # Update the total number of pixels.
            total_pixels += batch_size * h * w
            
            # Sum over all images in the batch (across height and width) for each channel.
            mean.add_(images.sum(dim=[0, 2, 3]))  # In-place addition
            # Sum of squares for each channel.
            std.add_((images ** 2).sum(dim=[0, 2, 3]))  # In-place addition
    
    # Compute the mean per channel.
    mean /= total_pixels
    # Compute variance and then std deviation.
    std = torch.sqrt(std / total_pixels - mean ** 2)
    
    return mean.cpu(), std.cpu()  # Move results back to CPU if needed
    
rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    h, w, _ = image_decoded.shape
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    #image_resized = image_resized.astype(dtype=np.float32)
    #image_resized = (image_resized / 127.5) - 1.0
    return image_resized, h, w

import json

class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=0, frame_step=1):
        self.dir = video_folder
        self.transform = transform
        self.video_frames = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self._frame_step = frame_step
        self.index_samples = []
        self.setup()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        videos.sort()
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.jpg'))
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            all_video_frames = videos
        
        self.video_frames = all_video_frames
        max_index = len(all_video_frames) - (self._time_step + self._num_pred - 1) * self._frame_step
        self.index_samples = list(range(max_index))
        #print('self.index_samples ***', self.index_samples)

    def __getitem__(self, index):
        seed = random.randint(0, 2**32 - 1)
        frame_index = self.index_samples[index]
        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))
        batch_of_frames= [] 

        for i in range(self._time_step + self._num_pred):
            torch.manual_seed(seed)
            frame_path = self.video_frames[frame_index + i * self._frame_step]
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_width, self._resize_height)
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)
            batch_of_frames.append(image)

            if self.transform is not None:
                image = (image).astype(np.uint8)  # Reverse normalization
                image = Image.fromarray(image)
                batch_frames[i] = self.transform(image)
        #batch_of_frames = [torch.tensor(frame) for frame in batch_of_frames]
        #batch_of_frames = torch.stack(batch_of_frames).permute(0,3,1,2)
        #batch_frames = self.transform(batch_of_frames)
        
        return {
            '256': batch_frames,
            'standard': batch_frames
        }

    def __len__(self):
        return len(self.index_samples)

