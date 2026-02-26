"""Clinical cardiac metrics: ejection fraction and myocardium wall thickness.

Myo thickness algorithm adapted from cmr-dinov3/myo_thickness.py (spoke-based
radial measurement). Volume/EF computed from voxel counts.
"""

import numpy as np
import scipy as sp
from shapely.geometry import LineString
from skimage.measure import find_contours

from data.util import SPACING

VOXEL_VOL_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0  # mm³ → mL



def compute_volumes(seg):
    """Compute chamber volumes in mL from integer label map.

    Args:
        seg: numpy array [D, H, W] with labels 0=BG, 1=RV, 2=Myo, 3=LV.

    Returns:
        dict with keys 'RV', 'Myo', 'LV' → volume in mL.
    """
    return {
        'RV': float(np.sum(seg == 1)) * VOXEL_VOL_ML,
        'Myo': float(np.sum(seg == 2)) * VOXEL_VOL_ML,
        'LV': float(np.sum(seg == 3)) * VOXEL_VOL_ML,
    }


def compute_ef(edv, esv):
    """Ejection fraction = (EDV - ESV) / EDV * 100.

    Returns float (percentage). Returns NaN if EDV <= 0.
    """
    if edv <= 0:
        return float('nan')
    return (edv - esv) / edv * 100.0



def _mask_to_contour(mask):
    """Binary 2D mask → (x, y) contour arrays."""
    contours = find_contours(mask.astype(np.uint8), level=0.5, fully_connected='high')
    if len(contours) == 0:
        return None, None
    c = contours[0]
    return c[:, 1], c[:, 0]  # x, y


def _interp_contour(cx, cy, npts=401):
    """Resample contour to npts equally-spaced (by arc length) points."""
    pts = np.stack([cx, cy], axis=1)
    idx = np.hstack([np.arange(len(cx)), 0])
    sample = pts[idx]
    d = np.diff(sample, axis=0)
    cum = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    cum = np.insert(cum, 0, 0)
    cum = np.cumsum(cum)
    cum = cum / cum[-1]
    t = np.linspace(0, 1, npts)
    ox = np.interp(t, cum, sample[:, 0], period=360)[:-1]
    oy = np.interp(t, cum, sample[:, 1], period=360)[:-1]
    return ox, oy


def _smooth_contour(cx, cy, n_freq=12, npts=401):
    """Fourier-smooth a contour via convex hull + low-pass filter."""
    pts = np.stack([cx, cy], axis=1)
    try:
        hull = sp.spatial.ConvexHull(pts)
        hv = np.hstack([hull.vertices, hull.vertices[0]])
        pts = pts[hv]
    except sp.spatial.QhullError:
        pass

    d = np.diff(pts, axis=0)
    cum = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    cum = np.insert(cum, 0, 0)
    cum = np.cumsum(cum)
    cum = cum / cum[-1]
    t = np.linspace(0, 1, npts)
    cx = np.interp(t, cum, pts[:, 0], period=360)[:-1]
    cy = np.interp(t, cum, pts[:, 1], period=360)[:-1]

    n = len(cx)
    nfilt = n - n_freq - 1
    for arr in [cx, cy]:
        f = np.fft.fft(arr)
        f[n // 2 + 1 - nfilt // 2:n // 2 + nfilt // 2] = 0.0
        arr[:] = np.real(np.fft.ifft(f))
    return cx, cy


def _order_pts_clockwise(xep, yep, xen, yen):
    """Reorder both contours to start from consistent origin, clockwise."""
    for xa, ya in [(xep, yep), (xen, yen)]:
        xmed, ymed = np.median(xa), np.median(ya)
        lhs = np.where(xa < xmed)[0]
        if len(lhs) == 0:
            ix = np.argmin(np.abs(ya - ymed))
        else:
            ix = lhs[np.argmin(np.abs(ya[lhs] - ymed))]
        xa[:] = np.hstack((xa[ix:], xa[:ix]))
        ya[:] = np.hstack((ya[ix:], ya[:ix]))

        x0, y0 = xa[0], ya[0]
        i90 = len(xa) // 4
        if ya[i90] > y0:
            xa[:] = xa[::-1]
            ya[:] = ya[::-1]
    return xep, yep, xen, yen


def _get_thickness(xen, yen, xep, yep):
    """Spoke-based wall thickness between endocardium and epicardium.

    Returns array of thickness values (one per endocardium point), in pixel units.
    """
    xep, yep, xen, yen = _order_pts_clockwise(xep, yep, xen, yen)

    xmi, ymi = np.mean(xen), np.mean(yen)
    xoc = np.hstack((xep, xep[0]))
    yoc = np.hstack((yep, yep[0]))
    epi_contour = LineString(np.vstack((xoc, yoc)).T)

    xtnd = 10
    thickness = np.zeros(len(xen))
    for ix in range(len(xen)):
        spoke = LineString(((xmi, ymi),
                            (xen[ix] + xtnd * (xen[ix] - xmi),
                             yen[ix] + xtnd * (yen[ix] - ymi))))
        isect = epi_contour.intersection(spoke)
        if isect.is_empty or isect.geom_type not in ('Point',):
            # Handle multi-point or no intersection
            if hasattr(isect, 'geoms'):
                # Pick the closest point to endo point
                dists = [np.hypot(p.x - xen[ix], p.y - yen[ix]) for p in isect.geoms]
                best = list(isect.geoms)[np.argmin(dists)]
                thickness[ix] = np.hypot(best.x - xen[ix], best.y - yen[ix])
            continue
        thickness[ix] = np.hypot(isect.x - xen[ix], isect.y - yen[ix])

    return thickness


def _myo_thickness_per_slice(endo_mask, epi_mask, pixel_spacing):
    """Mean wall thickness for one 2D slice in mm.

    Returns float or None if contours cannot be extracted.
    """
    xen, yen = _mask_to_contour(endo_mask)
    if xen is None or len(xen) < 10:
        return None
    xep, yep = _mask_to_contour(epi_mask)
    if xep is None or len(xep) < 10:
        return None

    xen, yen = _interp_contour(xen, yen)
    xen, yen = _smooth_contour(xen, yen)
    xep, yep = _interp_contour(xep, yep)
    xep, yep = _smooth_contour(xep, yep)

    try:
        t = _get_thickness(xen, yen, xep, yep)
    except Exception:
        return None

    t = t[t > 0]
    if len(t) == 0:
        return None
    return float(np.mean(t)) * pixel_spacing


def compute_myo_thickness(seg, pixel_spacing=1.5):
    """Mean LV myocardium wall thickness in mm across axial slices.

    Args:
        seg: numpy [D, H, W] integer labels (2=Myo, 3=LV).
        pixel_spacing: in-plane pixel spacing in mm (isotropic).

    Returns:
        float: mean wall thickness in mm, or NaN if no valid slices.
    """
    D = seg.shape[0]
    slice_means = []
    for d in range(D):
        s = seg[d]
        endo = (s == 3).astype(np.uint8)
        epi = ((s == 2) | (s == 3)).astype(np.uint8)
        if endo.sum() < 10 or epi.sum() < 10:
            continue
        t = _myo_thickness_per_slice(endo, epi, pixel_spacing)
        if t is not None:
            slice_means.append(t)
    if len(slice_means) == 0:
        return float('nan')
    return float(np.mean(slice_means))
