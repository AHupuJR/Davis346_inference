import torch
import numpy as np
import os


def events_to_accumulate_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), keep_middle=True):
    
    """
    to left: -
    to right: +
    ----
    --
     -
      -
       --
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_mid = ts[0] + (dt/2)
    
    # left of the mid -
    tend = t_mid
    end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
    for bi in range(int(B/2)):
        tstart = ts[0] + (dt/B)*bi
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(-vb) # !
    # self
    if keep_middle:
        bins.append(torch.zeros_like(vb))  # TODO!!!
    # right of the mid +
    tstart = t_mid
    beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
    for bi in range(int(B/2), B):
        tend = ts[0] + (dt/B)*(bi+1)
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(vb)

    bins = torch.stack(bins)
    return bins


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps, accumulate=True)
            # print("able to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
            #     ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img



def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search implemented for pytorch tensors (no native implementation exists)
    @param t The tensor
    @param x The value being searched for
    @param l Starting lower bound (0 if None is chosen)
    @param r Starting upper bound (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    @param pxs Numpy array of integer typecast x coords of events
    @param pys Numpy array of integer typecast y coords of events
    @param dxs Numpy array of residual difference between x coord and int(x coord)
    @param dys Numpy array of residual difference between y coord and int(y coord)
    @returns Image
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def voxel2mask(voxel):
    mask_final = np.zeros_like(voxel[0, :, :])
    mask = (voxel != 0)
    for i in range(mask.shape[0]):
        mask_final = np.logical_or(mask_final, mask[i, :, :])
    # to uint8 image
    mask_img = mask_final * np.ones_like(mask_final) * 255
    mask_img = mask_img[..., np.newaxis] # H,W,C
    mask_img = np.uint8(mask_img)

    return mask_img



class RobustNorm(object):
    """
    Robustly normalize tensor
    """

    def __init__(self, low_perc=0, top_perc=95):
        self.top_perc = top_perc
        self.low_perc = low_perc

    @staticmethod
    def percentile(t, q):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.

        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        try:
            result = t.view(-1).kthvalue(k).values.item()
        except RuntimeError:
            result = t.reshape(-1).kthvalue(k).values.item()
        return result

    def __call__(self, x, is_flow=False):
        """
        """
        t_max = self.percentile(x, self.top_perc)
        t_min = self.percentile(x, self.low_perc)
        # print("t_max={}, t_min={}".format(t_max, t_min))
        if t_max == 0 and t_min == 0:
            return x
        eps = 1e-6
        normed = torch.clamp(x, min=t_min, max=t_max)
        normed = normed / (abs(max(torch.min(normed), torch.max(normed), key=abs))+eps)
        
        return normed

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(top_perc={:.2f}'.format(self.top_perc)
        format_string += ', low_perc={:.2f})'.format(self.low_perc)
        return format_string
