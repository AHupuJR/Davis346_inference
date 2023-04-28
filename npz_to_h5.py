import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
# my packages
from model import EFNet
from event_utils import events_to_accumulate_voxel_torch, recursive_glob, voxel2mask, events_to_image_torch, RobustNorm
from model import load_network
import cv2
import h5py


# 输入参数
root_path = '/cluster/work/cvl/leisun/davis346/'
seq_name= 'blur-2023_03_22_10_10_46'
seq_path = os.path.join(root_path, seq_name)
# seq_path = '/cluster/work/cvl/leisun/davis346/blur-2023_03_22_10_10_46'
# img_path = '/cluster/work/cvl/leisun/davis346/blur-2023_03_22_10_10_46/1679451047868057.jpg'
# img_path = '/cluster/work/cvl/leisun/davis346/deblur-2023_04_10_16_10_30/deblur-2023_04_10_16_10_30/1681114230521808.jpg'
event_path = os.path.join(root_path, seq_name, seq_name+'.npy')

# event_path = '/cluster/work/cvl/leisun/davis346/blur-2023_03_22_10_10_46.npy'
# event_path = '/cluster/work/cvl/leisun/davis346/deblur-2023_04_10_16_10_30/deblur-2023_04_10_16_10_30.npy'

expo_sync_mode = 'start'
# weight_path = './net_g_latest_REBlur.pth'
expo_time = 25000
sensor_size = (260, 346) # -》 (256, 344)
num_bins = 6 # voxel channel number
### RobustNorm!
robustnorm = RobustNorm(low_perc=35, top_perc=65) # 95 85

# # 加载模型
# model = EFNet().cuda()
# # 测试跑通不需要权重
# # model.load_state_dict(torch.load(weight_path))
# model = load_network(model, weight_path, True, param_key='params')
# model.eval()

# data
img_paths = recursive_glob(rootdir=seq_path, suffix='.jpg')
input_events = np.load(event_path)
# shape in h5:
# img:chw np.ndarray, c=3
# voxel: chw np.ndarray
num_img=0

# output_h5_file_path = os.path.join(root_path, seq_name+'.h5')
output_h5_file_path = os.path.join(root_path, 'debug_test.h5')

print("saving file: {}".format(output_h5_file_path))
h5_file = h5py.File(output_h5_file_path,'a')

for img_path in img_paths:
    blur = cv2.imread(img_path)
    blur = torch.from_numpy(blur.transpose(2, 0, 1)) # chw
    ## Event 预处理
    ## filter the events that in expotime
    ts_img = int(os.path.basename(img_path).split('.')[0])
    if expo_sync_mode == 'mid':
        ts_img_start = ts_img - expo_time//2
        ts_img_end = ts_img + expo_time//2
    elif expo_sync_mode == 'start':
        ts_img_start = ts_img
        ts_img_end = ts_img + expo_time
    ## find events
    idxs = np.searchsorted(input_events[:,0], [ts_img_start, ts_img_end]) # 这一步没问题
    filtered_event = input_events[idxs[0]:idxs[1], :]
    # print(f'DEBUG: filtered_event:{filtered_event}')
    ts = filtered_event[:,0]
    # print(f'DEBUG: ts:{ts}')
    xs = filtered_event[:,1]
    ys = filtered_event[:,2]
    ps = filtered_event[:,3]

    ## events->voxel
    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))
    # ts = torch.from_numpy((ts-ts_0).astype(np.float32)) # ts start from 0
    ts = ts - ts[0]
    ts = torch.from_numpy(ts.astype(np.float32))
    ps = torch.from_numpy(ps.astype(np.float32))
    ps = ps*2. -1.# -1或1 只用正的画event分布图

    # import pdb; pdb.set_trace()
    ### DEBUG plot the event image
    # event_img = events_to_image_torch(xs, ys, ps,device=None, sensor_size=sensor_size, clip_out_of_range=True)
    
    # event_img /= abs(max(event_img.min(), event_img.max(), key=abs)) # normalize
    ###  changed to RobustNorm!!
    # event_img = robustnorm(event_img)
    # event_img = 255.0*event_img.numpy().astype(np.uint8)[..., np.newaxis] #hwc
    # cv2.imwrite(filename=os.path.join(root_path, 'robustnorm_event_visualize', 'event_{}.png'.format(num_img)), img=event_img)

    #####
    voxel = events_to_accumulate_voxel_torch(xs, ys, ts, ps, B=num_bins, sensor_size=sensor_size, keep_middle=False)
    voxel = robustnorm(voxel)
    voxel = voxel.numpy() # chw
    # voxel 可能有hot pixel! voxel的max和min太大了（100），不合理


    mask_img = voxel2mask(voxel)
    #close filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img_close = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # print(mask_img_close.shape)
    mask_img_close = mask_img_close[np.newaxis,...] # H,W -> C,H,W  C=1

    # save to h5 image
    voxel_dset=h5_file.create_dataset("voxels/voxel{:09d}".format(num_img), data=voxel, dtype=np.dtype(np.float32))
    image_dset=h5_file.create_dataset("images/image{:09d}".format(num_img), data=blur, dtype=np.dtype(np.uint8))
    mask_dset=h5_file.create_dataset("masks/mask{:09d}".format(num_img), data=mask_img_close, dtype=np.dtype(np.uint8))

    voxel_dset.attrs['size']=voxel.shape
    image_dset.attrs['size']=blur.shape
    mask_dset.attrs['size']=mask_img_close.shape
    image_dset.attrs['origin_name']=str(ts_img)



    num_img+=1
sensor_resolution = [blur.shape[1], blur.shape[2]]
h5_file.attrs['sensor_resolution'] = sensor_resolution


h5_file.close()







# # 进行推理
# with torch.no_grad():
#     output_tensor = model(x=input_tensor, event=voxel, mask=mask) # added mask
#     # print(f'DEBUG: output_tensor:{output_tensor}')

# output_tensor = output_tensor.cpu()
# output_image = output_tensor.squeeze().permute(1,2,0).clamp(0.0,1.0).numpy()
# output_image = (output_image*255.0).astype('uint8')

# # 保存输出图像
# output_image = Image.fromarray(output_image)

# output_image.save(result_path)
