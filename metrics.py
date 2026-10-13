import numpy as np
# intensity dismatch
# fixed [B,1,H,W]
# warpd [B,1,H,W]
def image_mse(fixed,warpd):
    fixed = np.asarray(fixed,dtype=np.float64)
    warpd = np.asarray(warpd,dtype=np.float64)
    if fixed.shape != warpd.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean((fixed-warpd)**2))
# anatomical overlap
def dice_for_label(fixed_labels,warpd_labels,label):
    fixed_labels = np.asarray(fixed_labels)
    warpd_labels = np.asarray(warpd_labels)
    if fixed_labels.shape != warpd_labels.shape:
        raise ValueError("Segmentation arrays must share shape.")
    fixed_region = fixed_labels ==label
    warpd_region = warpd_labels ==label
    intersection = fixed_region & warpd_region
    fixed_count = np.sum(fixed_region)
    warpd_count = np.sum(warpd_region)
    overlap_count = np.sum(intersection)
    total = fixed_count + warpd_count
    if total ==0: return np.nan
    return float(2*overlap_count/ total)
# dictionary of scores excluding background
def dice_per_label(fixed_labels,warpd_labels):
    labels = np.union1d(np.unique(fixed_labels),np.unique(warpd_labels))
    scores = {}
    for label in labels:
        if label != 0:
            scores[int(label)] = dice_for_label(fixed_labels,warpd_labels,label)
    return scores
# average foreground dice, give each label same weight
def mean_dice(fixed_labels,warpd_labels):
    scores = dice_per_label(fixed_labels,warpd_labels)
    if not scores:
        return float("nan")
    return float(np.mean(list(scores.values())))
# local expansion, compression
# whether learned deformation field is reasonable
# flow [2,H,W]
def jacobian_determinant(flow):
    flow = np.asarray(flow,dtype=np.float64)
    if flow.ndim != 3 or flow.shape[0] !=2:
        raise ValueError("Flow must have shape [2,H,W].")
    if min(flow.shape[1:]) < 2:
        raise ValueError("Height and width must be at least 2.")
    if not np.isfinite(flow).all():
        raise ValueError("Flow contains NaN or infinite values.")
    ux = flow[0]
    uy = flow[1]
    dux_dy, dux_dx = np.gradient(ux)
    duy_dy, duy_dx = np.gradient(uy)
    determinant = ((1.0+ dux_dx)*(1.0+duy_dy) - dux_dy * duy_dx)
    return determinant
# count locations with nonpositive determinant
# fraction of locations with nonpositive Jacobian determinant
def fold_fraction(flow):
    determinant = jacobian_determinant(flow)
    return float(np.mean(determinant <=0))
