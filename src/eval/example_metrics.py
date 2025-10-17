from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
import frechet_video_distance as fvd
from torchvision.models import inception_v3
from torcheval.metrics import FrechetInceptionDistance
import tensorflow.compat.v1 as tf
from PIL import Image
# Number of videos must be divisible by 16.
BATCH_SIZE, NUM_FRAMES = 16, 9

device='cuda:0'
# Create a session with GPU options
gpu_options = tf.GPUOptions(visible_device_list=f"0")
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True  # Dynamically grow the memory used on the GPU


# Load inception model for FID
inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# # # inception_model.load_state_dict(torch.load('pt_inception-2015-12-05-6726825d.pth'))
inception_model = inception_model.to(device)
inception_model.eval()
# # # Load a pre-trained Inception v3 model
# # inception_model = inception_v3(pretrained=True, transform_input=False)
# # inception_model.fc = torch.nn.Identity()  # Remove the last fully connected layer
# # inception_model = inception_model.eval().cuda()


# Load a pre-trained Inception v3 model
# inception_model = inception_v3(pretrained=True, transform_input=False)
# inception_model.fc = torch.nn.Identity()  # Remove the last fully connected layer
# inception_model = inception_model.eval().cuda()

# Transformation to resize and normalize images
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_frames(videos):
    """
    Preprocess the frames of the videos and return a tensor.
    """
    preprocessed_videos = []
    for video in videos:
        for frame in video:
            frame = Image.fromarray(frame)
            frame = preprocess(frame)
            preprocessed_videos.append(frame)
    return torch.stack(preprocessed_videos)

def get_features(frames, model):
    """
    Extract features from the frames using the Inception model.
    """
    features = []
    with torch.no_grad():
        for frame in frames:
            frame = frame.unsqueeze(0)
            feat = model(frame).cpu()
            features.append(feat)
    return torch.cat(features)

def preprocess_videos(videos):
    """
    Preprocess videos to [0, 1] and reshape to (batch_size * frames, channels, height, width).
    """
    videos = videos / 255.0  # Assuming input is [0, 255]
    batch_size, frames, channels, height, width = videos.shape
    videos = videos.reshape(batch_size * frames, channels, height, width)
    return videos



def calculate_fvd(generated_videos, real_videos):
    """
    Calculate FVD between predicted and ground truth videos.
    """
    with tf.Graph().as_default():
        real_videos = tf.convert_to_tensor(real_videos, np.float32) 
        generated_videos = tf.convert_to_tensor(generated_videos, np.float32) 

        result = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
            fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
        

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            fvd_score = sess.run(result)

    
    return fvd_score


def calculate_fid(predicted_videos, ground_truth_videos):
    # Transformation to resize and normalize images
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # # Preprocess and extract features for predicted and ground truth videos
    # predicted_videos = Image.fromarray(predicted_videos.reshape(-1, 64, 64, 3))
    # ground_truth_videos = Image.fromarray(ground_truth_videos.reshape(-1, 64, 64, 3))
    
    predicted_frames = torch.tensor(predicted_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64).to(device) / 256.0
    ground_truth_frames = torch.tensor(ground_truth_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64).to(device) / 256.0
    
    # predicted_frames = preprocess(torch.tensor(predicted_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64)).to(device)#.reshape(BATCH_SIZE, NUM_FRAMES, 3, 299, 299)
    # ground_truth_frames = preprocess(torch.tensor(ground_truth_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64)).to(device)#.reshape(BATCH_SIZE, NUM_FRAMES, 3, 299, 299)
    
    # predicted_frames = preprocess_frames((predicted_videos).permute(0, 1, 4, 2, 3))
    # ground_truth_frames = preprocess_frames((ground_truth_videos).permute(0, 1, 4, 2, 3))

    # predicted_features = get_features(predicted_frames, inception_model)
    # ground_truth_features = get_features(ground_truth_frames, inception_model)

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature_dim=2048, device=device)

    # Update FID metric with extracted features
    fid_metric.update(predicted_frames, is_real=False)
    fid_metric.update(ground_truth_frames, is_real=True)

    # Compute FID
    fid_value = fid_metric.compute()
    return fid_value

def calculate_mse(predicted_videos, ground_truth_videos):
    ground_truth_videos = torch.tensor(ground_truth_videos).float() / 256.0
    predicted_videos = torch.tensor(predicted_videos).float() / 256.0
    error_metric = torch.sqrt((ground_truth_videos[:, 1:] - predicted_videos[:, 1:])**2)
    mean_error  = error_metric.detach().cpu().numpy().mean()
    frame_acc_error = error_metric.sum(1).detach().cpu().numpy().mean()
    return mean_error, frame_acc_error
        

# Example usage
predicted_videos = np.ones((BATCH_SIZE, NUM_FRAMES, 64, 64, 3), dtype=np.uint8) * 255
ground_truth_videos = np.ones((BATCH_SIZE, NUM_FRAMES, 64, 64, 3), dtype=np.uint8) * 230

predicted_videos[14:] = 0
ground_truth_videos[14:] = 0

fvd_score = calculate_fvd(predicted_videos, ground_truth_videos)
fid_score = calculate_fid(predicted_videos, ground_truth_videos)
me, fe = calculate_mse(predicted_videos, ground_truth_videos)

print(f'FVD: {fvd_score}')
print(f'FID: {fid_score}')
print(f'MSE: {me}')
print(f'MFE: {fe}')



predicted_videos[12:] = 0
ground_truth_videos[12:] = 0

fvd_score = calculate_fvd(predicted_videos, ground_truth_videos)
fid_score = calculate_fid(predicted_videos, ground_truth_videos)
me, fe = calculate_mse(predicted_videos, ground_truth_videos)

print(f'FVD: {fvd_score}')
print(f'FID: {fid_score}')
print(f'MSE: {me}')
print(f'MFE: {fe}')



predicted_videos[8:] = 0
ground_truth_videos[8:] = 0

fvd_score = calculate_fvd(predicted_videos, ground_truth_videos)
fid_score = calculate_fid(predicted_videos, ground_truth_videos)
me, fe = calculate_mse(predicted_videos, ground_truth_videos)

print(f'FVD: {fvd_score}')
print(f'FID: {fid_score}')
print(f'MSE: {me}')
print(f'MFE: {fe}')



predicted_videos[6:] = 0
ground_truth_videos[6:] = 0

fvd_score = calculate_fvd(predicted_videos, ground_truth_videos)
fid_score = calculate_fid(predicted_videos, ground_truth_videos)
me, fe = calculate_mse(predicted_videos, ground_truth_videos)

print(f'FVD: {fvd_score}')
print(f'FID: {fid_score}')
print(f'MSE: {me}')
print(f'MFE: {fe}')

