import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.custom_fwd.*")

import os
os.environ['CUDA_LAUNCH_BLOCKING']="0"
import cv2
import numpy as np
import torch

from ted import TED
from utils.img_processing import image_normalization

_device = torch.device('cpu' if torch.cuda.device_count() == 0
                        else 'cuda')
if _device.type == 'cpu':
    print("Using cpu for TEED inference")

_model = TED().to(_device)
_model.eval()

_checkpoint_path = os.environ.get("TEED_CHECKPOINT", "path/to/your/teed_model.pth")
if not os.path.isfile(_checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint file not found: {_checkpoint_path}")
print(f"Loaded TEED.")
_model.load_state_dict(torch.load(_checkpoint_path,
                                    map_location=_device,
                                    weights_only=True))

_up_scale = True

def inference(_image):

    _image_shape = _image.shape[:2]
    _image_shape = (_image_shape[1], _image_shape[0])
    if _up_scale:

        _scale = 1080 / _image_shape[0]

        _image = cv2.resize(_image,(0,0),fx=_scale,fy=_scale)
    if _image.shape[0] < 512 or _image.shape[1] < 512:

        _image = cv2.resize(_image, (0, 0), fx=1.5, fy=1.5)

    if _image.shape[0] % 8 != 0 or _image.shape[1] % 8 != 0:
        img_width = ((_image.shape[1] // 8) + 1) * 8
        img_height = ((_image.shape[0] // 8) + 1) * 8
        _image = cv2.resize(_image, (img_width, img_height))

    else:
        pass
    _image = np.array(_image, dtype=np.float32)
    _mean_bgr = [103.939, 116.779, 123.68]
    _image -= _mean_bgr
    _image = _image.transpose((2, 0, 1))
    _image_tensor = torch.from_numpy(_image.copy()).float()
    _image_tensor = _image_tensor.unsqueeze(0)
    _image_tensor = _image_tensor.to(_device)

    with torch.no_grad():
        if _device.type == 'cuda':
            torch.cuda.synchronize()
        _preds = _model(_image_tensor)
        if _device.type == 'cuda':
            torch.cuda.synchronize()

        edge_maps = []
        for i in _preds:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)

        tmp = tensor[:, 0, ...]
        tmp = np.squeeze(tmp)

        fuse_num = tmp.shape[0]-1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)

            if not tmp_img.shape[1] == _image_shape[0] or not tmp_img.shape[0] == _image_shape[1]:
                tmp_img = cv2.resize(tmp_img, (_image_shape[0], _image_shape[1]))


            if i == fuse_num:

                fuse = tmp_img
                fuse = fuse.astype(np.uint8)

        torch.cuda.empty_cache()

    return fuse
