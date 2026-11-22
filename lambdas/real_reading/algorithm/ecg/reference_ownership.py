"""Give every measured stroke to the row it starts from, however far it goes.

Ink connected to one row belongs to that row without any window. Only where ink
joins rows are free tips followed along their straight measured strokes.
"""
import cv2
import numpy as np

SLOPES = np.arange(-.45, .4501, .0025, dtype=np.float32)


def follow_stroke(mask, x, y, direction, limit, avoid=None):
    """Longest straight measured support from a tip: (length, slope, offset)."""
    height, width = mask.shape
    ys = np.arange(y, y+direction*limit, direction)
    ys = ys[(ys >= 0) & (ys < height)]
    best = (0, 0., 0)
    if len(ys) < 5:
        return best
    for offset in (0, -1, 1):
        alive = np.arange(len(SLOPES))
        if avoid is not None:
            alive = alive[np.abs(SLOPES-avoid) > .08]
        done, size = 0, 16
        while alive.size:
            stop = min(len(ys), done+size)
            first = max(0, done-3)
            part = ys[first:stop]
            xs = np.rint(x+offset+SLOPES[alive, None]*(part-y)).astype(np.int32)
            inside = (xs >= 0) & (xs < width)
            miss = ~(mask[part, xs.clip(0, width-1)] & inside)
            # Up to three pixels may be lost at a printed-grid crossing.
            gap = miss[:, :-3] & miss[:, 1:-2] & miss[:, 2:-1] & miss[:, 3:]
            broken = gap.any(1)
            reach = first+gap.argmax(1) if gap.size else np.zeros(alive.size, int)
            if stop == len(ys):
                reach = np.where(broken, reach, len(ys)-1)
                broken[:] = True
            if broken.any():
                score = np.where(broken, reach, -1)
                # Several slopes fit a thick stroke: take the middle of them.
                tied = np.flatnonzero(score == score.max())
                k = int(tied[len(tied)//2])
                if reach[k] > best[0]:
                    best = (int(reach[k]), float(SLOPES[alive[k]]), offset)
            alive = alive[~broken]
            done, size = stop, size*2
    return best


def extend_back(mask, x, y, direction, slope, limit):
    """Measured continuation behind a tip along that same straight stroke."""
    height, width = mask.shape
    reached = misses = 0
    for step_back in range(1, limit):
        yy = int(y-direction*step_back)
        xx = int(round(x-direction*step_back*slope))
        if not (0 <= yy < height and 0 <= xx < width):
            break
        # Neighbouring rows are required too: a round cap is not a continuation.
        if mask[yy, xx]:
            reached, misses = step_back, 0
        else:
            misses += 1
            if misses > 1:
                break
    return reached if reached > 3 else 0


def free_ends(local, direction):
    """Pixels with no measured continuation above (1) or below (-1)."""
    padded = np.pad(local.astype(np.uint8), 4)
    if direction == -1:
        padded = padded[::-1]
    wide = cv2.dilate(padded, np.ones((1, 3), np.uint8))
    before = np.zeros_like(wide)
    for shift in (1, 2, 3):
        before[shift:] |= wide[:-shift]
    free = (padded > 0) & (before == 0)
    if direction == -1:
        free = free[::-1]
    free = free[4:-4, 4:-4]
    count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    for k in range(1, count):
        yy, xx = np.nonzero(labels == k)
        at = np.argmin(yy*direction)
        yield int(xx[at]), int(yy[at])


def resolve_origins(strokes, rows, step, tolerance=1.5):
    """Decide strokes that pass through several rows; None means undecidable."""
    # A piece lying on a longer measured stroke has that stroke's origin.
    def inside(short, long):
        low, high = sorted((long['y'], long['end']))
        for y, x in ((short['y'], short['x']),
                     (short['end'], short['x']+short['slope']*(short['end']-short['y']))):
            if not low-3 <= y <= high+3:
                return False
            if abs(long['x']+long['slope']*(y-long['y'])-x) > tolerance:
                return False
        return True
    crossing = [s for s in strokes if len(s['reach']) > 1]
    strokes[:] = [s for s in strokes if len(s['reach']) > 1
                  or not any(inside(s, long) for long in crossing)]
    for s in strokes:
        s['owner'] = s['reach'][0] if len(s['reach']) == 1 else None
    through = [s for s in strokes if len(s['reach']) > 1]

    def ends(s):
        tip = (s['x'], s['y'])
        far = (s['x']+s['slope']*(s['end']-s['y']), s['end'])
        return sorted((tip, far), key=lambda point: point[1])

    # A through stroke that stops exactly on a baseline starts from that row;
    # it cannot start from the row that it passes and leaves behind.
    for s in through:
        upper, lower = ends(s)
        first, last = s['reach'][0], s['reach'][-1]
        on_first = abs(upper[1]-rows[first]) <= .06*step
        on_last = abs(lower[1]-rows[last]) <= .06*step
        if on_first != on_last:
            s['owner'] = first if on_first else last
    # Two through strokes that meet in a tip are legs of one trace.
    changed = True
    while changed:
        changed = False
        for s in through:
            if s['owner'] is not None:
                continue
            found = set()
            for other in through:
                if other['owner'] not in s['reach']:
                    continue
                if any(abs(a[0]-b[0]) <= 4*tolerance and abs(a[1]-b[1]) <= .06*step
                       for a, b in zip(ends(s), ends(other))):
                    found.add(other['owner'])
            if len(found) == 1:
                s['owner'] = found.pop()
                changed = True
    for s in through:
        if s['owner'] is not None:
            continue
        direction, reach = s['direction'], s['reach']
        explained = set()
        for other in strokes:
            if other['direction'] != direction or len(other['reach']) != 1:
                continue
            row = other['reach'][0]
            if (rows[row]-other['y'])*direction > .15*step:
                explained.add(row)
        crossed = reach[0] if direction == 1 else reach[-1]
        origin = reach[-1] if direction == 1 else reach[0]
        remaining = [row for row in reach if row not in explained]
        # A row with its own tip on this side cannot own the second tip as well.
        if crossed in explained and len(remaining) == 1:
            s['owner'] = remaining[0]
        elif abs(s['end']-rows[origin]) <= .12*step:
            # The stroke turns into this baseline: that is where it started.
            s['owner'] = origin
        elif len(remaining) == 1:
            s['owner'] = remaining[0]
    return strokes


def inner_tips(mask, strokes, rows, step, stroke):
    """Tip of an excursion lying between the two legs of another row's peak.

    Both legs continue past it, so it has no free end outside. It ends where
    paper becomes visible again between the legs, measured on the photograph.
    """
    height, width = mask.shape
    found = []
    through = [s for s in strokes if len(s['reach']) == 2 and s['owner'] is not None]
    for pair in {tuple(s['reach']) for s in through}:
        for owner in pair:
            legs = [s for s in through if tuple(s['reach']) == pair and s['owner'] == owner]
            if len(legs) < 2:
                continue
            other = pair[0] if owner == pair[1] else pair[1]
            away = 1 if rows[owner] > rows[other] else -1
            level = rows[owner]-away*.1*step
            at = [s['x']+s['slope']*(level-s['y']) for s in legs]
            left, right = legs[int(np.argmin(at))], legs[int(np.argmax(at))]
            if max(at)-min(at) < stroke+2:
                continue
            solid, opened, tip = 0, False, None
            for y in range(int(level), int(rows[other]+away*.05*step), -away):
                a = left['x']+left['slope']*(y-left['y'])
                b = right['x']+right['slope']*(y-right['y'])
                if b-a < stroke or not 0 <= y < height:
                    break
                middle = int(round((a+b)/2))
                if not 0 <= middle < width:
                    break
                if mask[y, middle]:
                    solid += 1
                    # A printed grid line is thinner than an overlapped stroke.
                    if opened and solid >= .04*step:
                        tip = y+away*(solid-1)
                        break
                else:
                    solid, opened = 0, True
            if tip is not None:
                found.append(dict(row=other, owner=owner, away=away, y=int(tip),
                                  left=left, right=right))
    return found


def seed_components(mask, lengths, bases, step, pitch, stroke, reference):
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    labels[~mask] = 0
    flat = mask & (lengths <= max(3, round(2.2*stroke)))
    yy, xx = np.nonzero(flat)
    seeds = [[] for _ in range(count)]
    for row, base in enumerate(bases):
        # A trace starts inside the reference; it may then go anywhere.
        near = ((np.abs(yy-base[xx]) < .16*step)
                & (xx >= reference[0]) & (xx < reference[2]))
        pairs =np.unique(np.stack([labels[yy[near], xx[near]], xx[near]], 1), axis=0)
        support = np.bincount(pairs[:, 0], minlength=count)
        # A baseline is horizontal support; lettering is too narrow to seed it.
        for k in np.flatnonzero(support >= max(8, .025*pitch)):
            # A page border can cross a baseline but does not run along it.
            if (k and stats[k, 2] >= .06*pitch
                    and support[k] >= min(.15*pitch, .15*stats[k, 2])):
                seeds[k].append(row)
    # A trace has one value in time: smaller print under or over a traced
    # baseline (lettering, grid dashes) cannot start the same row again.
    covered = np.zeros((len(bases), mask.shape[1]), bool)
    for k in np.argsort(-stats[:, cv2.CC_STAT_AREA]):
        if not seeds[k]:
            continue
        x, y, w, h, _ = stats[k]
        present = (labels[y:y+h, x:x+w] == k).any(0)
        if len(seeds[k]) == 1 and w < .5*pitch:
            if covered[seeds[k][0], x:x+w][present].mean() >= .5:
                seeds[k] = []
                continue
        for row in seeds[k]:
            covered[row, x:x+w] |= present
    return labels, stats, seeds


def split_component(mask, local, origin, rows_in, bases, step, stroke):
    """Return one measured mask per row, plus pixels whose origin is undecidable."""
    left, top = origin
    height, width = local.shape
    local_bases = bases[:, left:left+width]-top
    ys = np.arange(height, dtype=np.float32)[:, None]
    default = np.full(local.shape, -1, np.int16)
    distance = np.full(local.shape, np.inf, np.float32)
    for row in rows_in:
        near = np.abs(ys-local_bases[row])
        closer = local & (near < distance)
        default[closer], distance[closer] = row, near[closer]
    links = []
    margin = .12*step
    for upper, lower in zip(rows_in, rows_in[1:]):
        band = local & (ys > local_bases[upper]+margin) & (ys < local_bases[lower]-margin)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
        for k in range(1, count):
            x, y, w, h, _ = stats[k]
            part = labels[y:y+h, x:x+w] == k
            yy, xx = np.nonzero(part)
            at_top = bool((yy+y <= local_bases[upper, xx+x]+margin+3).any())
            at_bottom = bool((yy+y >= local_bases[lower, xx+x]-margin-3).any())
            selected = default[y:y+h, x:x+w]
            if at_top and at_bottom:
                links.append((x, x+w, y, part))
            elif at_top:
                selected[part] = upper
            elif at_bottom:
                selected[part] = lower
    owned = {row: default == row for row in rows_in}
    disputed = np.zeros(local.shape, bool)
    if not links:
        return owned, disputed, []
    all_strokes = []
    painted = {row: np.zeros(local.shape, bool) for row in rows_in}
    unresolved = np.zeros(local.shape, bool)
    linked = np.zeros(local.shape, bool)
    reach_margin = int(max(10, .03*step))
    radius = int(np.ceil(stroke/2))+1
    spans = []
    for start, stop, y, part in sorted(links, key=lambda link: link[0]):
        linked[y:y+part.shape[0], start:stop] |= part
        # A slanted stroke reaches its tip beyond the columns where it links rows.
        margin_x = max(reach_margin, min(stop-start, int(.5*step)))
        xa, xb = max(0, start-margin_x), min(width, stop+margin_x)
        if spans and xa <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], xb)
        else:
            spans.append([xa, xb])
    # A one-pixel stroke is staircased: allow its immediate neighbour when probing.
    probe = mask if stroke >= 2 else cv2.dilate(mask.astype(np.uint8), np.ones((1, 3), np.uint8)) > 0
    for xa, xb in spans:
        rows = bases[:, left+(xa+xb)//2]
        strokes = []
        for direction in (1, -1):
            for xx, yy in free_ends(local[:, xa:xb], direction):
                tip_x, tip_y = left+xa+xx, top+yy
                avoid = None
                # A tip joins two legs; follow each of them when they differ.
                for _ in range(2):
                    length, slope, offset = follow_stroke(
                        probe, tip_x, tip_y, direction, height+4, avoid)
                    if length < .1*step:
                        break
                    avoid = slope
                    end = tip_y+direction*length
                    # The measured stroke may begin before this apparent tip.
                    back = extend_back(probe, tip_x+offset, tip_y, direction, slope, height+4)
                    x, y_tip = tip_x-direction*back*slope, tip_y-direction*back
                    low, high = sorted((y_tip, end))
                    reach = [r for r in rows_in if low-.05*step <= rows[r] <= high+.05*step]
                    if reach:
                        strokes.append(dict(x=x+offset, y=y_tip, end=end, slope=slope,
                                            direction=direction, reach=reach))
        resolve_origins(strokes, rows, step, 1.5 if stroke >= 2 else 2.5)
        inner = inner_tips(mask, strokes, rows, step, stroke)
        for tip in inner:
            # The overlapped excursion is shared ink up to its measured tip.
            row, a, b = tip['row'], tip['left'], tip['right']
            low, high = sorted((int(rows[row]), tip['y']))
            low, high = max(low, top), min(high, top+height-1)
            for y in range(low, high+1):
                edges = sorted((a['x']+a['slope']*(y-a['y']), b['x']+b['slope']*(y-b['y'])))
                first, last = max(0, int(edges[0]-left)), min(width, int(edges[1]-left)+1)
                painted[row][y-top, first:last] |= local[y-top, first:last]
        for s in strokes:
            target = unresolved if s['owner'] is None else painted[s['owner']]
            begin, finish = s['y'], s['end']
            for tip in inner:
                if tip['row'] != s['owner'] or abs(s['x']-tip['left']['x']) > .15*step:
                    continue
                # It cannot continue along the other row's legs beyond that tip.
                limit = min if tip['away'] == 1 else max
                begin, finish = limit(begin, tip['y']), limit(finish, tip['y'])
            if begin == finish:
                continue
            canvas = np.zeros(local.shape, np.uint8)
            first = (int(s['x']+s['slope']*(begin-s['y'])-left), int(begin-top))
            last = (int(round(s['x']+s['slope']*(finish-s['y'])-left)), int(finish-top))
            cv2.line(canvas, first, last, 1, 2*radius+1)
            target |= (canvas > 0) & local
        all_strokes.extend(strokes)
    anyone = np.logical_or.reduce(list(painted.values()))
    for row in rows_in:
        # A stroke of another origin leaves this row; a true crossing stays in both.
        owned[row] = painted[row] | (owned[row] & ~anyone & ~linked & ~unresolved)
    disputed = (unresolved | linked) & ~anyone
    return owned, disputed, all_strokes


def assign_rows(mask, lengths, bases, step, pitch, stroke, reference):
    labels, stats, seeds = seed_components(mask, lengths, bases, step, pitch, stroke, reference)
    owned = np.zeros((len(bases), *mask.shape), bool)
    disputed = np.zeros(mask.shape, bool)
    decisions = []
    for k, rows_in in enumerate(seeds):
        if not rows_in:
            continue
        x, y, w, h, _ = stats[k]
        local = labels[y:y+h, x:x+w] == k
        if len(rows_in) == 1:
            owned[rows_in[0], y:y+h, x:x+w] |= local
            continue
        parts, unclear, strokes = split_component(mask, local, (x, y), rows_in, bases, step, stroke)
        for row, part in parts.items():
            owned[row, y:y+h, x:x+w] |= part
        disputed[y:y+h, x:x+w] |= unclear
        decisions.extend(strokes)
    attach_fragments(labels, stats, seeds, owned, bases, stroke)
    attach_continuations(labels, stats, seeds, owned, bases, step, stroke, reference)
    return owned, disputed, decisions


def attach_fragments(labels, stats, seeds, owned, bases, stroke):
    """A stroke broken at a grid crossing rejoins the stroke that it continues."""
    height, width = labels.shape
    gap = int(np.ceil(3*stroke))
    side = int(np.ceil(stroke))
    pending = [k for k in range(1, len(stats))
               if not seeds[k] and stats[k, 3] >= 3*stroke and stats[k, 3] >= 2*stats[k, 2]]
    for _ in range(3):
        waiting = []
        for k in pending:
            x, y, w, h, _ = stats[k]
            local = labels[y:y+h, x:x+w] == k
            found = set()
            for row_y, top, bottom in ((0, y-gap, y), (h-1, y+h, y+h+gap)):
                xs = np.flatnonzero(local[row_y])
                if not xs.size:
                    continue
                a, b = max(0, x+xs[0]-side), min(width, x+xs[-1]+side+1)
                top, bottom = max(0, top), min(height, bottom)
                if bottom > top and b > a:
                    found.update(np.flatnonzero(owned[:, top:bottom, a:b].any((1, 2))).tolist())
            if len(found) == 1:
                owned[found.pop(), y:y+h, x:x+w] |= local
            else:
                waiting.append(k)
        if len(waiting) == len(pending):
            break
        pending = waiting


def attach_continuations(labels, stats, seeds, owned, bases, step, stroke, reference):
    """A baseline broken outside the reference continues to its measured end."""
    width = labels.shape[1]
    gap = int(np.ceil(4*stroke))
    slack = 2*stroke
    middle = (reference[0]+reference[2])/2
    candidates = [k for k in range(1, len(stats))
                  if not seeds[k] and stats[k, 2] >= 2*stroke
                  and (stats[k, 0] < reference[0] or stats[k, 0]+stats[k, 2] > reference[2])]
    # Grow outwards from the reference, one touching piece at a time.
    candidates.sort(key=lambda k: abs(stats[k, 0]+stats[k, 2]/2-middle))
    for k in candidates:
        x, y, w, h, _ = stats[k]
        local = labels[y:y+h, x:x+w] == k
        rightwards = x+w/2 > middle
        inner = x if rightwards else x+w-1
        ys = np.flatnonzero(local[:, inner-x])
        if not ys.size:
            continue
        level = y+float(np.median(ys))
        for row, base in enumerate(bases):
            if abs(level-base[inner]) > .16*step:
                continue
            if rightwards:
                a, b = max(0, inner-gap), inner
            else:
                a, b = inner+1, min(width, inner+gap+1)
            top = max(0, int(level-slack))
            if b <= a or not owned[row, top:int(level+slack)+1, a:b].any():
                continue
            # It must extend the row, not duplicate columns already traced.
            low = max(0, int(base[inner]-.3*step))
            traced = owned[row, low:int(base[inner]+.3*step), x:x+w].any(0)
            if traced.mean() < .3:
                owned[row, y:y+h, x:x+w] |= local
            break


def assign_sections(row_mask, lengths, base, boundaries, step, stroke):
    """Cut at the marker on the baseline; a detached excursion keeps its section."""
    height, width = row_mask.shape
    column = np.searchsorted(boundaries[1:], np.arange(width), side='right').astype(np.int16)
    result = np.where(row_mask, column[None, :], -1).astype(np.int16)
    if len(boundaries) < 2:
        return result
    cut = row_mask.copy()
    half = max(1, int(np.ceil(stroke/2)))
    for edge in boundaries[1:]:
        x = int(round(edge))
        if not 0 <= x < width:
            continue
        low = max(0, int(base[x]-.18*step))
        high = min(height, int(base[x]+.18*step))
        cut[low:high, max(0, x-half):x+half+1] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), 8)
    for k in range(1, count):
        x, y, w, h, _ = stats[k]
        span = column[x:x+w]
        if span[0] == span[-1]:
            continue
        local = labels[y:y+h, x:x+w] == k
        yy, xx = np.nonzero(local & (lengths[y:y+h, x:x+w] <= 2.2*stroke))
        body = np.abs(yy+y-base[xx+x]) < .16*step
        votes = np.bincount(span[xx[body]], minlength=len(boundaries))
        present = np.flatnonzero(votes >= max(3, 2*stroke))
        if len(present) == 1:
            result[y:y+h, x:x+w][local] = present[0]
    return result


def release_labels(ink, owned, markers, step, stroke):
    """Lettering beside a section marker is not signal, even when it nearly touches.

    Only whole detached glyphs just under the marker are released: a wave that
    runs through a label is never cut.
    """
    height, width = ink.shape
    found = []
    for x, y, w, h in markers:
        left = max(0, int(x-.02*step))
        right = min(width, int(x+.4*step))
        top = max(0, int(y+h-.1*step))
        bottom = min(height, int(y+h+.25*step))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink[top:bottom, left:right].astype(np.uint8), 8)
        for k in range(1, count):
            a, b, wide, tall, area = stats[k]
            if (a and b and a+wide < right-left and b+tall < bottom-top
                    and .03*step <= tall <= .15*step and wide <= .15*step
                    and area >= 3*stroke and top+b >= y+h-.03*step):
                selected = labels[b:b+tall, a:a+wide] == k
                owned[:, top+b:top+b+tall, left+a:left+a+wide][:, selected] = False
                found.append([int(left+a), int(top+b), int(wide), int(tall)])
        glyphs = [g for g in found if left <= g[0] < right and top <= g[1] < bottom]
        if len(glyphs) < 2:
            continue
        # A last letter may be fused to the wave. Letters share one printed
        # bottom line; only ink that ends exactly on that line is lettering.
        line = int(np.median([g[1]+g[3] for g in glyphs]))
        first = int(min(g[1] for g in glyphs))+1
        a = min(g[0]+g[2] for g in glyphs)
        b = min(width, int(max(g[0]+g[2] for g in glyphs)+.12*step))
        if line-2 <= first or b <= a:
            continue
        for part in owned[:, first:min(height, line+3), a:b]:
            below = part[line-first+1:].any(0)
            ends = part[line-first-2:line-first+1].any(0) & ~below
            lettering = cv2.dilate(ends.astype(np.uint8)[None], np.ones((1, 13), np.uint8))[0] > 0
            part[:, lettering & ~below] = False
    return found
