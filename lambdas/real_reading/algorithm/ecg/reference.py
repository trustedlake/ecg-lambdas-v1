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


def measure_time_scale(image, box):
    """Position of each printed major vertical line, inside and outside the box.

    On curved paper the squares are not equally wide. The returned knots make
    them equal, keeping the average square measured inside the reference.
    """
    x0, y0, x1, y1 = box
    rgb = image[y0:y1].astype(np.int16)
    warmth = (rgb[..., 0]-rgb[..., 1:].min(2)).mean(0).astype(np.float32)
    detail = np.maximum(0, warmth-cv2.GaussianBlur(warmth.reshape(1, -1), (0, 0), 8).ravel())
    inside = detail[x0:x1]-detail[x0:x1].mean()
    size = len(inside)
    spectrum = np.fft.rfft(inside, 2*size)
    auto = np.fft.irfft(np.abs(spectrum)**2, 2*size)[:size]
    if auto[0] <= 0:
        return None
    auto = auto/auto[0]
    low, high = max(12, int(size*.012)), int(size*.05)  # one large square
    if high <= low+2:
        return None
    period = float(low+np.argmax(auto[low:high]))
    if auto[int(period)] < .25:
        return None
    middle = (x0+x1)//2
    half = int(period/2)
    first = middle-half+int(np.argmax(detail[middle-half:middle+half]))
    minimum = .35*float(np.percentile(detail[x0:x1], 99))

    def walk(direction):
        at, pitch, lost, count = float(first), period, 0, 0
        while True:
            count += 1
            expected = at+direction*pitch
            reach = max(2, int(.15*period))
            a, b = int(expected-reach), int(expected+reach)+1
            if a < 0 or b > len(detail):
                return
            part = detail[a:b]
            if part.max() < minimum:
                lost += 1
                at = expected
                if lost > 1:
                    return
                continue
            k = a+int(part.argmax())
            around = detail[max(0, k-2):k+3]
            found = float(np.dot(around, np.arange(max(0, k-2), max(0, k-2)+len(around)))/around.sum())
            if not lost:
                pitch = .6*pitch+.4*abs(found-at)
            at, lost = found, 0
            yield direction*count, found

    lines = sorted([(0, float(first)), *walk(-1), *walk(1)])
    index, positions = np.asarray(lines, float).T
    if len(lines) < 15 or positions[0] > x0+.1*(x1-x0) or positions[-1] < x1-.1*(x1-x0):
        return None
    if np.any(np.diff(positions) <= 0):
        return None
    degree = 4 if len(lines) >= 25 else 2
    centre, spread = float(index.mean()), float(np.ptp(index))
    smooth = np.polyfit((index-centre)/spread, positions, degree)
    error = np.abs(np.polyval(smooth, (index-centre)/spread)-positions)
    if float(np.median(error)) > .06*period:
        return None
    regular = np.arange(index[0], index[-1]+1)
    old = np.polyval(smooth, (regular-centre)/spread)
    steps = np.diff(old)
    if steps.min() < .6*period or steps.max() > 1.6*period:
        return None
    start, stop = np.interp([x0, x1], old, regular)
    if x0 < old[0]:
        start = regular[0]+(x0-old[0])/steps[0]
    if x1 > old[-1]:
        stop = regular[-1]+(x1-old[-1])/steps[-1]
    square = (x1-x0)/(stop-start)
    new = (regular-start)*square+x0
    last = image.shape[1]-1
    # Beyond the last printed line the last measured square is kept.
    if old[0] > 0:
        new = np.r_[new[0]-old[0]*square/steps[0], new]
        old = np.r_[0., old]
    if old[-1] < last:
        new = np.r_[new, new[-1]+(last-old[-1])*square/steps[-1]]
        old = np.r_[old, float(last)]
    # Nothing of the photograph may fall off the left or right side.
    new = new-new[0]
    return {'working_x': new.tolist(), 'before_x': old.tolist(),
            'major_square_px': float(square), 'measured_lines': len(lines),
            'square_min_px': float(steps.min()), 'square_max_px': float(steps.max()),
            'median_fit_error_px': float(np.median(error))}


def equalize_squares(image, scale):
    height = image.shape[0]
    new, old = np.asarray(scale['working_x']), np.asarray(scale['before_x'])
    width = int(np.ceil(new[-1]))+1
    columns = np.interp(np.arange(width), new, old).astype(np.float32)
    result = np.empty((height, width, 3), image.dtype)
    for top in range(0, height, 256):
        stop = min(height, top+256)
        map_x = np.repeat(columns[None, :], stop-top, axis=0)
        map_y = np.repeat(np.arange(stop-top, dtype=np.float32)[:, None], width, axis=1)
        result[top:stop] = cv2.remap(image[top:stop], map_x, map_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    return result


def _level_terms(u, v):
    return [u, u**2, u**3, u**4, u*v, u**2*v, u**3*v, u*v**2, u**2*v**2]


def measure_level(image, box, square):
    """Vertical bend of the printed horizontal lines, followed strip by strip."""
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    span, tall = x1-x0, y1-y0
    left, right = max(0, int(x0-.1*span)), min(width, int(x1+.1*span))
    top, bottom = max(0, int(y0-.3*tall)), min(height, int(y1+.3*tall))
    strip = max(16, int(square*1.5))
    edges = np.arange(left, right-strip+1, strip)
    if len(edges) < 6:
        return None
    profiles = []
    for a in edges:
        rgb = image[top:bottom, a:a+strip].astype(np.int16)
        warmth = (rgb[..., 0]-rgb[..., 1:].min(2)).mean(1).astype(np.float32)
        smooth = cv2.GaussianBlur(warmth.reshape(-1, 1), (0, 0), 8).ravel()
        profiles.append(np.maximum(0, warmth-smooth))
    centres = edges+(strip-1)/2
    middle = int(np.abs(centres-(x0+x1)/2).argmin())
    reach, window = max(3, int(square*.3)), int(square*5)
    shifts = np.arange(-reach, reach+1)
    samples = []
    for centre in np.linspace(top+window/2+reach, bottom-window/2-reach-1, 8):
        for direction in (1, -1):
            k, moved = middle, 0.
            while 0 <= k+direction < len(edges):
                a = int(round(centre+moved-window/2))-top
                if a < reach or a+window+reach >= bottom-top:
                    break
                base = profiles[k][a:a+window]
                base = base-base.mean()
                scores = []
                for shift in shifts:
                    other = profiles[k+direction][a+shift:a+shift+window]
                    other = other-other.mean()
                    norm = float(np.linalg.norm(base)*np.linalg.norm(other))
                    scores.append(float(base@other)/norm if norm else 0.)
                best = int(np.argmax(scores))
                # A lost grid is never extrapolated strip by strip.
                if scores[best] < .45 or best in (0, len(shifts)-1):
                    break
                below, at, above = scores[best-1:best+2]
                curve = below-2*at+above
                moved += shifts[best]+(.5*(below-above)/curve if curve else 0.)
                k += direction
                samples.append((centres[k], centre, moved))
    if len(samples) < 40:
        return None
    x, y, shift = np.asarray(samples, float).T
    if np.ptp(x) < .7*span:
        return None
    mid_x, mid_y = float(centres[middle]), float((y0+y1)/2)
    matrix = np.stack(_level_terms((x-mid_x)/span, (y-mid_y)/tall), 1)
    keep = np.ones(len(x), bool)
    for _ in range(4):
        values = np.linalg.lstsq(matrix[keep], shift[keep], rcond=None)[0]
        error = np.abs(matrix@values-shift)
        keep = error <= max(2., 3*float(np.median(error)))
        if keep.sum() < 40:
            return None
    if float(np.median(error[keep])) > .06*square:
        return None
    return {'centre': [mid_x, mid_y], 'size': [float(span), float(tall)],
            'u_range': [float((x.min()-mid_x)/span), float((x.max()-mid_x)/span)],
            'v_range': [float((y.min()-mid_y)/tall), float((y.max()-mid_y)/tall)],
            'terms': values.tolist(), 'samples': int(keep.sum()),
            'median_fit_error_px': float(np.median(error[keep]))}


def level_shift(level, x, y):
    u = np.clip((x-level['centre'][0])/level['size'][0], *level['u_range'])
    v = np.clip((y-level['centre'][1])/level['size'][1], *level['v_range'])
    return sum(value*term for value, term in zip(level['terms'], _level_terms(u, v)))


def level_lines(image, level):
    """Make printed horizontal lines straight, without removing any row."""
    height, width = image.shape[:2]
    result = np.empty_like(image)
    columns = np.arange(width, dtype=np.float32)[None, :]
    for top in range(0, height, 256):
        rows = np.arange(top, min(height, top+256), dtype=np.float32)[:, None]
        map_y = (rows+level_shift(level, columns, rows)).astype(np.float32)
        first = max(0, int(np.floor(map_y.min()))-2)
        stop = min(height, max(first+1, int(np.ceil(map_y.max()))+3))
        result[top:top+len(rows)] = cv2.remap(image[first:stop], np.repeat(columns, len(rows), 0),
                                              map_y-first, cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)
    return result


def working_to_source(geometry, x, y):
    """Map traced points back to the untouched photograph."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    level = geometry.get('level')
    if level:
        y = y+level_shift(level, x, y)
    scale = geometry.get('time_scale')
    if scale:
        x = np.interp(x, scale['working_x'], scale['before_x'])
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
    time_scale = level = None
    if tilt:
        image = upright_verticals(image, tilt)
        time_scale = measure_time_scale(image, bounds)
    if time_scale:
        image = equalize_squares(image, time_scale)
        left, right = np.interp([x0, x1], time_scale['before_x'], time_scale['working_x'])
        bounds = [round(float(left)), y0, round(float(right)), y1]
        nw = image.shape[1]
        level = measure_level(image, bounds, time_scale['major_square_px'])
        if level:
            image = level_lines(image, level)
    geometry = dict(method='explicit_reference_full_photo', vertical_tilt=tilt,
                    time_scale=time_scale, level=level,
                    source_size=[sw, sh],
                    working_size=[nw, nh], reference_box_px=bounds,
                    source_coordinate_system='decoded_EXIF_oriented_pixels',
                    source_to_working=transform.tolist(),
                    working_to_source=np.linalg.inv(transform).tolist(),
                    orientation={'rotation_clockwise': rotation_clockwise, 'uncertain': False},
                    model='explicit_rotation_and_scale',
                    nonlinear_paper_warp_corrected=('both_axes' if level else
                                                    'time_axis_only' if time_scale else False))
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
    scale = geometry.get('time_scale')
    if scale and grid['valid']:
        count = config.get('major_interval_minor_squares', 5)
        spacing = scale['major_square_px']/count
        measured = spacing/grid['minor_square_mm']
        # Every line was followed across the sheet: finer than the tiled estimate.
        if abs(measured/grid['x']['pixels_per_mm']-1) <= .03:
            grid['x'].update(spacing_px=spacing, pixels_per_mm=measured,
                             source='followed_vertical_lines')
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
