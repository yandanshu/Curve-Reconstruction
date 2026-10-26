import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

import cv2
import numpy as np
import torch
from PIL import Image

from romatch import roma_outdoor

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    _device = torch.device('mps')

_COARSE_RES = 560
_DEFAULT_UPSAMPLE_RES = (640, 640)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    _roma_model = roma_outdoor(
        device=_device,
        weights=torch.load(os.environ["ROMA_WEIGHTS"], map_location=_device, weights_only=True)
        if os.environ.get("ROMA_WEIGHTS") else None,
        dinov2_weights=torch.load(os.environ["DINOV2_WEIGHTS"], map_location=_device, weights_only=True)
        if os.environ.get("DINOV2_WEIGHTS") else None,
        coarse_res=_COARSE_RES,
        upsample_res=_DEFAULT_UPSAMPLE_RES,
    )
print("Loaded RoMa.")

def inference(input_im1, input_im2):
    H1, W1 = input_im1.shape[:2]
    H2, W2 = input_im2.shape[:2]

    W_exp = 640.0
    H_exp = H1 * W_exp / W1
    H_ds = int(round(H_exp))
    W_ds = int(round(W_exp))

    im1_ds = Image.fromarray(cv2.resize(input_im1, (W_ds, H_ds)))
    im2_ds = Image.fromarray(cv2.resize(input_im2, (W_ds, H_ds)))

    upsample_res = (H_ds, W_ds)
    if _roma_model.upsample_res != upsample_res:
        _roma_model.upsample_res = upsample_res
    H_ds, W_ds = _roma_model.get_output_resolution()

    with torch.no_grad():
        warp, certainty = _roma_model.match(im1_ds, im2_ds, device=_device)

    warp_np = warp.detach().cpu().numpy()
    certainty_np = certainty.detach().cpu().numpy()

    map12_ds = warp_np[:, :W_ds, 2:]
    map21_ds = warp_np[:, W_ds:, :2]
    certainty12_ds = certainty_np[:, :W_ds]
    certainty21_ds = certainty_np[:, W_ds:]

    map12 = np.zeros((H1, W1, 2), dtype=map12_ds.dtype)
    map21 = np.zeros((H2, W2, 2), dtype=map21_ds.dtype)
    for c in range(2):
        map12[:, :, c] = cv2.resize(
            map12_ds[:, :, c], (W1, H1), interpolation=cv2.INTER_LINEAR)
        map21[:, :, c] = cv2.resize(
            map21_ds[:, :, c], (W2, H2), interpolation=cv2.INTER_LINEAR)

    certainty12 = cv2.resize(
        certainty12_ds, (W1, H1), interpolation=cv2.INTER_LINEAR)
    certainty21 = cv2.resize(
        certainty21_ds, (W2, H2), interpolation=cv2.INTER_LINEAR)

    if _device.type == 'cuda':
        torch.cuda.empty_cache()

    return map12, map21, certainty12, certainty21
