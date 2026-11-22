"""Explicit reference geometry; the image is never cropped for extraction.

Coordinates are in the decoded, EXIF-oriented source. Rotation is explicit and
the reference rectangle belongs to the rotated frame. No source identity enters
these operations. This is a local algorithm interface, not an upload contract.
"""
from copy import deepcopy
import cv2
import numpy as np
from PIL import Image, ImageOps
from .grid import estimate_grid


def measure_vertical_tilt(image, box):
    """dx/dy of printed vertical grid lines, as a linear function of x.

    Measured from long coloured straight lines around the reference. Without
    enough evidence nothing is corrected.
    """
    x0, y0, x1, y1 = box
    margin = .25*(y1-y0)
    top, bottom = max(0, int(y0-margin)), min(image.shape[0], int(y1+margin))
    rgb = image[top:bottom, x0:x1].astype(np.int16)
    warmth = rgb[..., 0]-rgb[..., 1:].min(2)
    printed = (warmth > np.median(warmth)+20).astype(np.uint8)*255
    # A printed grid is sparse; a texture that fills the area is not a grid.
    if not .002 < np.count_nonzero(printed)/printed.size < .35:
        return None
    length = max(40, int((y1-y0)*.08))
    lines = cv2.HoughLinesP(printed, 1, np.pi/720, length//2,
                            minLineLength=length, maxLineGap=4)
    if lines is None:
        return None
    a, b, c, d = lines.reshape(-1, 4).T.astype(float)
    dy = d-b
    steep = np.abs(c-a) < .18*np.abs(dy)
    if steep.sum() < 20:
        return None
    x, slope = (a+c)[steep]/2+x0, (c-a)[steep]/dy[steep]
    weight = np.sqrt(np.abs(dy[steep]))
    if np.ptp(x) < .5*(x1-x0):
        return None
    keep = np.ones(len(x), bool)
    for _ in range(4):
        linear = np.polyfit(x[keep], slope[keep], 1, w=weight[keep])
        error = np.abs(slope-np.polyval(linear, x))
        keep = error <= max(.006, 3*float(np.median(error)))
        if keep.sum() < 20:
            return None
    if float(np.median(error[keep])) > .012:
        return None
    return {'slope_per_px': float(linear[0]), 'slope_at_zero': float(linear[1]),
            'pivot_y_px': float((y0+y1)/2), 'measured_lines': int(keep.sum())}


def upright_verticals(image, tilt):
    """Make the time axis vertical, retaining the whole photograph."""
    height, width = image.shape[:2]
    result = np.empty_like(image)
    columns = np.arange(width, dtype=np.float32)
    slopes = tilt['slope_at_zero']+tilt['slope_per_px']*columns
    for top in range(0, height, 256):
        ys = np.arange(top, min(height, top+256), dtype=np.float32)
        map_x = (columns+slopes*(ys[:, None]-tilt['pivot_y_px'])).astype(np.float32)
        map_y = np.repeat((ys-top)[:, None], width, axis=1)
        result[top:top+len(ys)] = cv2.remap(image[top:top+len(ys)], map_x, map_y,
                                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result


def working_to_source(geometry, x, y):
    """Map traced points back to the untouched photograph."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    tilt = geometry.get('vertical_tilt')
    if tilt:
        x = x+(tilt['slope_at_zero']+tilt['slope_per_px']*x)*(y-tilt['pivot_y_px'])
    m = np.asarray(geometry['working_to_source'])
    return m[0, 0]*x+m[0, 1]*y+m[0, 2], m[1, 0]*x+m[1, 1]*y+m[1, 2]


def prepare_reference(source, config, rectangle, rotation_clockwise=0):
    if rotation_clockwise not in (0, 90, 180, 270):
        raise ValueError('Explicit right-angle rotation required')
    box = np.asarray(rectangle, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box < 0) or np.any(box > 1)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('Invalid normalized reference rectangle')
    with Image.open(source) as original:
        rgb = ImageOps.exif_transpose(original).convert('RGB')
    sw, sh = rgb.size
    transforms = {
        0: np.eye(3),
        90: np.array([[0, -1, sh-1], [1, 0, 0], [0, 0, 1]]),
        180: np.array([[-1, 0, sw-1], [0, -1, sh-1], [0, 0, 1]]),
        270: np.array([[0, 1, 0], [-1, 0, sw-1], [0, 0, 1]])}
    image = np.rot90(np.asarray(rgb), -(rotation_clockwise//90)).copy()
    h, w = image.shape[:2]
    # Keep the same trace resolution as a crop decoded with max_side. The full
    # photograph may be larger; this avoids shrinking thin ink a second time.
    reference_side = max((box[2]-box[0])*w, (box[3]-box[1])*h)
    scale = min(1., float(config.get('max_side', 2400))/reference_side)
    nw, nh = round(w*scale), round(h*scale)
    if (nw, nh) != (w, h):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = nw/w, nh/h
    resize = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    transform = resize @ transforms[rotation_clockwise]
    bounds = np.rint(box * [nw, nh, nw, nh]).astype(int).tolist()
    x0, y0, x1, y1 = bounds
    if min(x1-x0, y1-y0) < 100:
        raise ValueError('Reference too small to measure the grid')
    tilt = measure_vertical_tilt(image, bounds)
    if tilt:
        image = upright_verticals(image, tilt)
    geometry = dict(method='explicit_reference_full_photo', vertical_tilt=tilt,
                    source_size=[sw, sh],
                    working_size=[nw, nh], reference_box_px=bounds,
                    source_coordinate_system='decoded_EXIF_oriented_pixels',
                    source_to_working=transform.tolist(),
                    working_to_source=np.linalg.inv(transform).tolist(),
                    orientation={'rotation_clockwise': rotation_clockwise, 'uncertain': False},
                    model='explicit_rotation_and_scale', nonlinear_paper_warp_corrected=False)
    return image, geometry


def reference_grid(image, config, geometry):
    """Measure within the reference, and independently check adjoining paper.

    The exterior checks are evidence, not a claim of a curved-paper correction.
    Inconsistent local scale is explicitly reported for review.
    """
    x0, y0, x1, y1 = geometry['reference_box_px']
    grid = estimate_grid(image[y0:y1, x0:x1], config)
    grid['reference_box_px'] = [x0, y0, x1, y1]
    for axis, offset in (('x', x0), ('y', y0)):
        if grid[axis].get('phase_px') is not None:
            grid[axis]['phase_px'] += offset
        if grid['controls'].get(axis) is not None:
            grid['controls'][axis] = [v+offset for v in grid['controls'][axis]]
    h, w = image.shape[:2]
    checks = []
    for bounds in ([x0, max(0, y0-(y1-y0)), x1, y0],
                   [x0, y1, x1, min(h, y1+(y1-y0))],
                   [max(0, x0-(x1-x0)), y0, x0, y1],
                   [x1, y0, min(w, x1+(x1-x0)), y1]):
        a, b, c, d = bounds
        if min(c-a, d-b) < 100:
            continue
        local_config = deepcopy(config)
        local_config['grid_width_mm_hint'] *= (c-a)/(x1-x0)
        measured = estimate_grid(image[b:d, a:c], local_config)
        checks.append({'box_px': bounds, 'valid': measured['valid'],
                       'x': measured['x'], 'y': measured['y']})
    grid['exterior_checks'] = checks
    grid['local_scale_consistent'] = bool(grid['valid'] and checks and all(
        abs(check[a]['pixels_per_mm']/grid[a]['pixels_per_mm']-1) <= .08
        for check in checks if check['valid'] for a in ('x', 'y'))
        and any(check['valid'] for check in checks))
    return grid
