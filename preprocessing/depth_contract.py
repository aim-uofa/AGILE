"""Validity contract for the uint16 millimetre depth consumed by AGILE."""
import numpy as np


def alignment_depth_mask(depth_m, hand_mask, mano_mask):
    """Reject missing and saturated depth before constructing ICP correspondences."""
    depth_m = np.asarray(depth_m)
    hand_mask = np.asarray(hand_mask)
    mano_mask = np.asarray(mano_mask)
    if depth_m.shape != hand_mask.shape or depth_m.shape != mano_mask.shape:
        raise ValueError("Depth and hand masks must have identical shapes")
    # 65535 is also the legacy exporter's clipping value for infinity.
    return ((hand_mask > 0) & (mano_mask > 0) & np.isfinite(depth_m)
            & (depth_m > 0) & (depth_m < 65.535))
