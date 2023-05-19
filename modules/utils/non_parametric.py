import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as tr_F

def masked_average_pooling(mask_b1hw, feature_bchw, normalization):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Assert remove mask is not in mask provided
    assert -1 not in mask_b1hw

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    
    if normalization:
        feature_norm = torch.norm(feature_bchw, p=2, dim=1).unsqueeze(1).expand_as(feature_bchw)
        feature_bchw = feature_bchw.div(feature_norm + 1e-5) # avoid div by zero

    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)

def crop_partial_img(img_chw, mask_hw, cls_id=1):
    if isinstance(mask_hw, np.ndarray):
        mask_hw = torch.tensor(mask_hw)
    binary_mask_hw = (mask_hw == cls_id)
    binary_mask_hw_np = binary_mask_hw.numpy().astype(np.uint8)
    # RETR_EXTERNAL to keep online the outer contour
    contours, _ = cv2.findContours(binary_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop annotated objects off the image
    # Compute a minimum rectangle containing the object
    assert len(contours) != 0
    cnt = contours[0]
    x_min = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
    x_max = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
    y_min = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
    y_max = tuple(cnt[cnt[:,:,1].argmax()][0])[1]
    for cnt in contours:
        x_min = min(x_min, tuple(cnt[cnt[:,:,0].argmin()][0])[0])
        x_max = max(x_max, tuple(cnt[cnt[:,:,0].argmax()][0])[0])
        y_min = min(y_min, tuple(cnt[cnt[:,:,1].argmin()][0])[1])
        y_max = max(y_max, tuple(cnt[cnt[:,:,1].argmax()][0])[1])
    # Index of max bounding rect are inclusive so need 1 offset
    x_max += 1
    y_max += 1
    # mask_roi is a boolean arrays
    mask_roi = binary_mask_hw[y_min:y_max,x_min:x_max]
    img_roi = img_chw[:,y_min:y_max,x_min:x_max]
    return (img_roi, mask_roi)

def copy_and_paste(novel_img_chw, novel_mask_hw, base_img_chw, base_mask_hw, mask_id):
    base_img_chw = base_img_chw.clone()
    base_mask_hw = base_mask_hw.clone()
    # Horizontal Flipping
    if torch.rand(1) < 0.5:
        novel_img_chw = tr_F.hflip(novel_img_chw)
        novel_mask_hw = tr_F.hflip(novel_mask_hw)
    
    # Parameters for random resizing
    scale = np.random.uniform(0.1, 2.0)
    src_h, src_w = novel_mask_hw.shape
    if src_h * scale > base_mask_hw.shape[0]:
        scale = base_mask_hw.shape[0] / src_h
    if src_w * scale > base_mask_hw.shape[1]:
        scale = base_mask_hw.shape[1] / src_w
    target_H = int(src_h * scale)
    target_W = int(src_w * scale)
    if target_H == 0: target_H = 1
    if target_W == 0: target_W = 1
    # apply
    novel_img_chw = tr_F.resize(novel_img_chw, (target_H, target_W))
    novel_mask_hw = novel_mask_hw.view((1,) + novel_mask_hw.shape)
    novel_mask_hw = tr_F.resize(novel_mask_hw, (target_H, target_W), interpolation=tv.transforms.InterpolationMode.NEAREST)
    novel_mask_hw = novel_mask_hw.view(novel_mask_hw.shape[1:])

    # Random Translation
    h, w = novel_mask_hw.shape
    if base_mask_hw.shape[0] > h and base_mask_hw.shape[1] > w:
        paste_x = torch.randint(low=0, high=base_mask_hw.shape[1] - w, size=(1,))
        paste_y = torch.randint(low=0, high=base_mask_hw.shape[0] - h, size=(1,))
    else:
        paste_x = 0
        paste_y = 0
    
    base_img_chw[:,paste_y:paste_y+h,paste_x:paste_x+w][:,novel_mask_hw] = novel_img_chw[:,novel_mask_hw]
    base_mask_hw[paste_y:paste_y+h,paste_x:paste_x+w][novel_mask_hw] = mask_id

    img_chw = base_img_chw
    mask_hw = base_mask_hw

    return (img_chw, mask_hw)

def semantic_seg_CRF(pred_bhw):
    raise NotImplementedError
