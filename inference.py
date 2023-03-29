import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
# my packages
from model import EFNet
from event_utils import events_to_accumulate_voxel_torch
from model import load_network


# 输入参数
img_path = '/cluster/work/cvl/leisun/davis346/blur-2023_03_22_10_10_46/1679451047868057.jpg'
event_path = '/cluster/work/cvl/leisun/davis346/blur-2023_03_22_10_10_46.npy'
expo_sync_mode = 'start'
# weight_path = './net_g_latest_REBlur.pth'
weight_path = './net_g_latest_GoPro.pth'
result_path = '/cluster/work/cvl/leisun/davis346/result.png'
expo_time = 25000
sensor_size = (260, 346) # -》 (256, 344)
num_bins = 6 # voxel channel number


# 加载模型
model = EFNet().cuda()
# 测试跑通不需要权重
# model.load_state_dict(torch.load(weight_path))
model = load_network(model, weight_path, True, param_key='params')

model.eval()

### 数据读取
input_image = Image.open(img_path)
input_events = np.load(event_path)
# print(f'DEBUG: input_events:{input_events}')

## image 预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(input_image).unsqueeze(0).cuda()
## 1channel -> 3channels n c h w
input_tensor = torch.cat([input_tensor,input_tensor,input_tensor], dim=1)


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
idxs = np.searchsorted(input_events[:,0], [ts_img_start, ts_img_end])
filtered_event = input_events[idxs[0]:idxs[1], :]
# print(f'DEBUG: filtered_event:{filtered_event}')

ts = filtered_event[:,0]
# print(f'DEBUG: ts:{ts}')
xs = filtered_event[:,1]
ys = filtered_event[:,2]
ps = filtered_event[:,3]

## events->voxel
xs = torch.from_numpy(xs.astype(np.float32)).cuda()
ys = torch.from_numpy(ys.astype(np.float32)).cuda()
# ts = torch.from_numpy((ts-ts_0).astype(np.float32)) # ts start from 0
ts = ts - ts[0]
ts = torch.from_numpy(ts.astype(np.float32)).cuda() # !
ps = torch.from_numpy(ps.astype(np.float32)).cuda()
ps = ps*2. -1.

# print(f'DEBUG: xs:{xs}')
# print(f'DEBUG: ys:{ys}')
# print(f'DEBUG: ts:{ts}')
# print(f'DEBUG: ps:{ps}')

voxel = events_to_accumulate_voxel_torch(xs, ys, ts, ps, B=num_bins, sensor_size=sensor_size, keep_middle=False)
# voxel 没问题
# print(f'DEBUG: voxel:{voxel}')

# normalize
voxel = voxel / abs(max(voxel.min(), voxel.max(), key=abs))  # -1 ~ 1
# voxel = voxel * abs(max(voxel.min(), voxel.max(), key=abs))
print(f'voxel.max:{voxel.max()}')
print(f'voxel.min:{voxel.min()}')

voxel = voxel.unsqueeze(0)
# print(f'DEBUG: voxel after normalize:{voxel}')

# 截取至(256, 344)
input_tensor = input_tensor[:,:, :256, :344]
# print(f'DEBUG: input_tensor:{input_tensor}')

voxel = voxel[:,:, :256, :344]
# print(f'DEBUG: voxel:{voxel}')

# voxel *= 100000000000000000.
# print(f'DEBUG: voxel:{voxel}')

# 进行推理
with torch.no_grad():
    output_tensor = model(x=input_tensor, event=voxel)
    # print(f'DEBUG: output_tensor:{output_tensor}')

output_tensor = output_tensor.cpu()
output_image = output_tensor.squeeze().permute(1,2,0).clamp(0.0,1.0).numpy()
output_image = (output_image*255.0).astype('uint8')

# 保存输出图像
output_image = Image.fromarray(output_image)

output_image.save(result_path)
