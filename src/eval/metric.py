from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
import frechet_video_distance as fvd
from torchvision.models import inception_v3
from torcheval.metrics import FrechetInceptionDistance
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# import tensorflow as tf
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
# from metrics.calculate_fvd import calculate_fvd
from metrics.calculate_psnr import calculate_psnr
from metrics.calculate_ssim import calculate_ssim
from metrics.calculate_lpips import calculate_lpips


gpu=0
F, C, H, W = 9, 3, 64, 64  # Number of videos must be divisible by 16.
exp_name = sys.argv[1] if len(sys.argv) >= 2 else 'gem' 
visu_name = sys.argv[2] if len(sys.argv) >= 3 else 'visu'
# visu_idx = int(sys.argv[2]) if len(sys.argv) >= 3 else 6
gpu = int(sys.argv[3]) if len(sys.argv) >=4 else 0


device=f'cuda:0'
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"  # Use GPU 0
os.environ["LD_LIBRARY_PATH"]="/vision-nfs/torralba/env/yichenl/miniforge3/envs/tfeval/lib/"
# Configure TensorFlow to use only the specified GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

gpu_options = tf.GPUOptions(visible_device_list=f"0")
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True  # Dynamically grow the memory used on the GPU


def calculate_fvd(generated_videos, real_videos):
    """
    Calculate FVD between predicted and ground truth videos.
    """
    # with tf.device(f'/device:GPU:{gpu}'):
        
        # real_videos = tf.convert_to_tensor(real_videos, np.float32) 
        # generated_videos = tf.convert_to_tensor(generated_videos, np.float32) 

        # result = fvd.calculate_fvd(
        #     fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
        #     fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
        
        
        # fvd_score = result.numpy()
    

    # with tf.Session(config=config) as sess:
    #     real_videos = tf.convert_to_tensor(real_videos, np.float32) 
    #     generated_videos = tf.convert_to_tensor(generated_videos, np.float32) 
        
    #     result = fvd.calculate_fvd(
    #         fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
    #         fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
    
    #     fvd_score = sess.run(result)
    
    with tf.Graph().as_default():
        real_videos = tf.convert_to_tensor(real_videos, np.float32) 
        generated_videos = tf.convert_to_tensor(generated_videos, np.float32) 

        result = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
            fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
    

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
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
    
    
    predicted_frames = torch.tensor(predicted_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64).to(device) #/ 256.0
    ground_truth_frames = torch.tensor(ground_truth_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64).to(device) #/ 256.0
    
    # predicted_frames = preprocess(torch.tensor(predicted_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64)).to(device)#.reshape(BATCH_SIZE, NUM_FRAMES, 3, 299, 299)
    # ground_truth_frames = preprocess(torch.tensor(ground_truth_videos).float().permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64)).to(device)#.reshape(BATCH_SIZE, NUM_FRAMES, 3, 299, 299)
    
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature_dim=2048, device=device)

    # Update FID metric with extracted features
    fid_metric.update(predicted_frames, is_real=False)
    fid_metric.update(ground_truth_frames, is_real=True)

    # Compute FID
    fid_value = fid_metric.compute()
    fid_value = fid_value.detach().cpu().numpy()
    return fid_value


def calculate_mse(predicted_videos, ground_truth_videos):
    predicted_videos = torch.tensor(predicted_videos).float().permute(0, 1, 4, 2, 3) #/ 256.0
    ground_truth_videos = torch.tensor(ground_truth_videos).float().permute(0, 1, 4, 2, 3) #/ 256.0
    
    error_metric = torch.sqrt((ground_truth_videos[:, 1:] - predicted_videos[:, 1:])**2)
    mean_error  = error_metric.detach().cpu().numpy().mean()
    frame_acc_error = error_metric.sum(1).detach().cpu().numpy().mean()
    return mean_error, frame_acc_error
        
        
if __name__ == '__main__':
    exp_dir = os.path.join('trained_ckpt', exp_name)
    print(exp_dir)
    results_dir = glob(exp_dir + f'/{visu_name}')[0]
    all_results = glob(results_dir+'/metric/*.npy')
    all_results = sorted(all_results)
    all_fvd = []
    all_fid = []
    all_mse = []
    all_mfe = []
    all_ssim = []
    all_psnr = []
    all_lpips = []
    
    f = open(results_dir + '/metric/metrics.txt', 'w')
    
    gt_indices = np.arange(32).reshape(-1, 2)[:, 0]
    pred_indices = np.arange(32).reshape(-1, 2)[:, 1]

    
    B = np.load(all_results[0]).shape[0] // 2
    total_videos = B * len(all_results)
    total_round = total_videos // 16
    
    results = np.load(all_results[0])
    counter = 1
    round_iter = 0
    while counter < len(all_results):
        while results.shape[0] < 32 and counter < len(all_results):
            results = np.concatenate([results, np.load(all_results[counter])], axis=0)
            counter += 1
        
        current_results = results[:32]
        remaining_video = results[32:] if results.shape[0] > 32 else np.array([]).reshape(0, F, C, H, W)
        results = remaining_video
        
        B, F, C, H, W = current_results.shape
        gt_results = current_results[gt_indices]
        pred_results = current_results[pred_indices] 
        
        fvd_score = calculate_fvd(pred_results * 255, gt_results * 255)
        fid_score = calculate_fid(pred_results, gt_results)
        me, fe = calculate_mse(pred_results, gt_results)
        
        ssim_score = calculate_ssim(pred_results, gt_results)
        psnr_score = calculate_psnr(pred_results, gt_results)
        lpips_score = calculate_lpips(pred_results, gt_results, device)

        all_fvd.append(fvd_score)
        all_fid.append(fid_score)
        all_mse.append(me)
        all_mfe.append(fe)
        all_ssim.append((ssim_score['value'], ssim_score['value_std']))
        all_psnr.append((psnr_score['value'], psnr_score['value_std']))
        all_lpips.append((lpips_score['value'], lpips_score['value_std']))
        
        
        print(f'{round_iter} {counter} {fvd_score} {fid_score} {me} {fe}')
        f.write(f'{round_iter} {counter} {fvd_score} {fid_score} {me} {fe} \n')
        round_iter += 1
    
    fvds=np.array(all_fvd)
    fid=np.array(all_fid)
    mse=np.array(all_mse)
    mfe=np.array(all_mfe)
    ssim=np.array(all_ssim)
    psnr=np.array(all_psnr)
    lpipss=np.array(all_lpips)
    
    print(exp_name, ':')
    print(f'FVD: {np.mean(fvds)}')
    print(f'FID: {np.mean(fid)}')
    print(f'MSE: {np.mean(me)}')
    print(f'MFE: {np.mean(fe)}')
    print(f'SSIM: {np.mean(ssim[:, 0])} {np.mean(ssim[:, 1])}')
    print(f'PSNR: {np.mean(psnr[:, 0])} {np.mean(psnr[:, 1])}')
    print(f'lpips: {np.mean(lpipss[:,0])} {np.mean(lpipss[:, 1])}')
    

    np.savez(results_dir + '/metric/eval.npz', fvd=fvds, fid=fid, mse=mse, mfe=mfe, ssim=ssim, psnr=psnr, lpips=lpipss)
    
    f.close()
    
    with open(f'./{exp_name}_{visu_idx}.txt', 'w') as f:
        f.write(f'FVD: {np.mean(fvds)} \n')
        f.write(f'FVD: {np.mean(fvds)} \n')
        f.write(f'FID: {np.mean(fid)} \n')
        f.write(f'MSE: {np.mean(me)} \n')
        f.write(f'MFE: {np.mean(fe)} \n')
        f.write(f'SSIM: {np.mean(ssim[:, 0])} {np.mean(ssim[:, 1])} \n')
        f.write(f'PSNR: {np.mean(psnr[:, 0])} {np.mean(psnr[:, 1])} \n')
        f.write(f'lpips: {np.mean(lpipss[:,0])} {np.mean(lpipss[:, 1])} \n')
    
    # for i, respath in enumerate(tqdm(all_results)): 
    #     result = np.load(respath)
    #     B, F, C, H, W = result.shape

    #     while B < 32:
    #         result = np.concatenate([result, result], axis=0)
    #         B = B * 2

    #     gt_indices = np.arange(B).reshape(-1, 2)[:, 0]
    #     pred_indices = np.arange(B).reshape(-1, 2)[:, 1]
    #     gt_results = result[gt_indices]
    #     pred_results = result[pred_indices] 
    #     gt_results = gt_results[:16]
    #     pred_results = pred_results[:16]
        
    #     fvd_score = calculate_fvd(pred_results, gt_results)
    #     fid_score = calculate_fid(pred_results, gt_results)
    #     me, fe = calculate_mse(pred_results, gt_results)
        
    #     all_fvd.append(fvd_score)
    #     all_fid.append(fid_score)
    #     all_mse.append(me)
    #     all_mfe.append(fe)
        
    #     f.write(f'{i} {respath.split('.')[0].split('_')[1]} {fvd_score} {fid_score} {me} {fe} \n')
        

    # print(f'FVD: {fvd_score}')
    # print(f'FID: {fid_score}')
    # print(f'MSE: {me}')
    # print(f'MFE: {fe}')

