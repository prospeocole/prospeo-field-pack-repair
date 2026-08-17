"""Raster repairs for the v2 individual pack PNGs (one-pager, banner, cover).
All geometry measured from the gold at runtime; partner content only moves."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JBMB = '/home/claude/fonts/JetBrainsMono-Bold.ttf'

def die(msg): raise SystemExit('PNG GUARD FAILED: ' + msg)

def arr(im): return np.array(im.convert('RGB'))

def ink_mask(a, thresh=200):
    return a.max(axis=2) < thresh

def dark_neutral(a, lum=140, spread=46):
    mx, mn = a.max(axis=2).astype(int), a.min(axis=2).astype(int)
    return (mx < lum) & ((mx - mn) < spread)

def green_mask(a, ref=None, tol=None):
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    return (g > 88) & (g > r + 25) & (g > b + 12) & (r < 190)

def row_bands(mask, min_rows=3, min_px=8, gap=6):
    prof = mask.sum(axis=1)
    rows = np.where(prof >= min_px)[0]
    if len(rows) == 0: return []
    bands, s, p = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - p > gap:
            bands.append((s, p)); s = r
        p = r
    bands.append((s, p))
    return [(a, b) for a, b in bands if b - a + 1 >= min_rows]

def col_groups(mask, band, x0, x1, gap_frac=0.38):
    sub = mask[band[0]:band[1] + 1, x0:x1]
    prof = sub.sum(axis=0)
    cols = np.where(prof > 0)[0]
    if len(cols) == 0: return []
    h = band[1] - band[0] + 1
    gap = max(6, int(h * gap_frac))
    groups, s, p = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - p > gap:
            groups.append((x0 + s, x0 + p)); s = c
        p = c
    groups.append((x0 + s, x0 + p))
    return groups

def blob_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) < 50: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def circular(photo_path, size):
    im = Image.open(photo_path).convert('RGB').resize((size, size), Image.LANCZOS)
    m = Image.new('L', (size * 4, size * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    im.putalpha(m.resize((size, size), Image.LANCZOS))
    return im

def grad_fill(im, a, x0, y0, x1, y1, gap=10, depth=16):
    """Fill rect with background reconstructed per column from clear rows
    above (preferred) or below, so gradients stay exact."""
    H, W = a.shape[:2]
    ya0, ya1 = max(0, y0 - gap - depth), max(1, y0 - gap)
    yb0, yb1 = min(H - 1, y1 + gap), min(H, y1 + gap + depth)
    px = im.load()
    for x in range(max(0, x0), min(W, x1 + 1)):
        above = a[ya0:ya1, x].astype(int)
        below = a[yb0:yb1, x].astype(int)
        col = None
        if len(above) and above.std(axis=0).mean() < 6:
            col = np.median(above, axis=0)
        elif len(below) and below.std(axis=0).mean() < 6:
            col = np.median(below, axis=0)
        elif len(above):
            col = np.median(above, axis=0)
        c = tuple(int(v) for v in col)
        for y in range(max(0, y0), min(H, y1 + 1)):
            px[x, y] = c

def sample_bg(a, x, y, r=6):
    patch = a[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
    return tuple(int(v) for v in np.median(patch, axis=0))

def photo_state(a, bbox, headshot):
    """'match' own headshot, 'disc' flat avatar, 'empty' background, else fail."""
    x0, y0, x1, y1 = bbox
    reg = a[y0:y1 + 1, x0:x1 + 1]
    cx0, cy0 = reg.shape[1] // 4, reg.shape[0] // 4
    core = reg[cy0:-cy0 or None, cx0:-cx0 or None].reshape(-1, 3)
    med = np.median(core, axis=0)
    close = (np.abs(core - med).max(axis=1) < 16).mean()
    if close > 0.85 and med.min() > 240: return 'empty'
    if close > 0.55 and med.max() < 245: return 'disc'
    hs = arr(Image.open(headshot).resize((reg.shape[1], reg.shape[0])))
    if np.abs(hs.astype(int) - reg.astype(int)).mean() < 60: return 'match'
    return 'photo-unknown'

def paste_photo(im, bbox, headshot, cover_disc=False):
    x0, y0, x1, y1 = bbox
    if cover_disc:
        a = arr(im)
        pad = 70
        ry0, ry1 = max(0, y0 - pad), min(a.shape[0], y1 + pad)
        rx0, rx1 = max(0, x0 - pad), min(a.shape[1], x1 + pad)
        reg = a[ry0:ry1, rx0:rx1]
        cx, cy = (x1 - x0) // 2 + x0 - rx0, (y1 - y0) // 2 + y0 - ry0
        med = np.median(reg[max(0, cy - 20):cy + 20, max(0, cx - 20):cx + 20].reshape(-1, 3), axis=0)
        mask = np.abs(reg.astype(int) - med).max(axis=2) < 24
        if mask.sum() > 1500:
            bb = blob_bbox(mask)
            if bb:
                x0 = min(x0, rx0 + bb[0]); y0 = min(y0, ry0 + bb[1])
                x1 = max(x1, rx0 + bb[2]); y1 = max(y1, ry0 + bb[3])
    d = max(x1 - x0, y1 - y0) + 7
    ctr = ((x0 + x1) // 2, (y0 + y1) // 2)
    c = circular(headshot, d)
    im.paste(c, (ctr[0] - d // 2, ctr[1] - d // 2), c)

def strip_tail_words(im, a, band, x0, x1, n_expected, n_drop, region_note, mask_fn=None):
    """Erase the last n_drop word groups in the band. Returns kept-right-edge."""
    m = (mask_fn or dark_neutral)(a)
    groups = None
    for gf in (0.38, 0.30, 0.24, 0.19, 0.15):
        g_ = col_groups(m, band, x0, x1, gap_frac=gf)
        if len(g_) == n_expected:
            groups = g_; break
    if groups is None:
        die(f'{region_note}: expected {n_expected} word groups, found {len(g_)} at finest split: {g_}')
    keep_x1 = groups[-n_drop - 1][1]
    wipe_x0 = groups[-n_drop][0]
    grad_fill(im, a, wipe_x0 - 3, band[0] - 8, groups[-1][1] + 6, band[1] + 10)
    return keep_x1, groups

def move_region(im, a, bbox, dx, dy, bg, pad=11):
    x0, y0, x1, y1 = bbox
    crop = im.crop((x0 - pad, y0 - pad, x1 + 1 + pad, y1 + 1 + pad))
    ImageDraw.Draw(im).rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg)
    im.paste(crop, (x0 - pad + dx, y0 - pad + dy))


def last_rule_y(a, y_below, note):
    """The rule directly under the bottom-most bold left-column table label."""
    dm = strict_dark(a[:y_below, 60:700])
    bands = [b for b in row_bands(dm, min_rows=18, min_px=30)]
    if not bands: die(f'{note}: no table label band above y{y_below}')
    return int(bands[-1][1])

def wipe_stray_rules(im, a, y0, y1):
    mx, mn = a.max(axis=2).astype(int), a.min(axis=2).astype(int)
    rm = (mx < 246) & (mn > 195) & ((mx - mn) < 26)
    prof = rm.sum(axis=1)
    rows = np.where(prof > 1200)[0]
    rows = rows[(rows > y0) & (rows < y1)]
    for r in rows:
        ImageDraw.Draw(im).rectangle((90, int(r) - 2, 2290, int(r) + 2), fill=(255, 255, 255))
    return len(rows)

def trailing_group(bands, max_gap=45):
    if not bands: return []
    grp = [bands[-1]]
    for b in reversed(bands[:-1]):
        if grp[0][0] - b[1] < max_gap:
            grp.insert(0, b)
        else:
            break
    return grp

def color_mask(a, ref, tol=26):
    d = np.abs(a.astype(int) - np.array(ref)).max(axis=2)
    return d < tol

def strict_dark(a, mx=80):
    return a.max(axis=2) < mx

GOLD_OP_SB = (2053, 341, 2255, 361)
GOLD_OP_NAME = (1985, 383, 2257, 417)
GOLD_OP_PHOTO = (2056, 110, 2261, 313)

def _trailing_cluster(cols, min_gap=60):
    if len(cols) == 0: return None
    splits = np.where(np.diff(cols) > min_gap)[0]
    start = cols[splits[-1] + 1] if len(splits) else cols[0]
    return int(start), int(cols[-1])

def header_block(a, note, x_lo=1300, y_hi=620, need_photo=False):
    """Locate SHARED BY and name via right-aligned trailing column clusters."""
    sub = a[:y_hi, x_lo:]
    sbm = color_mask(sub, (124, 139, 153), 30)
    sb_bbox = None
    for b in row_bands(sbm, min_rows=5, min_px=12):
        if not (8 <= b[1] - b[0] <= 45): continue
        cols = np.where(sbm[b[0]:b[1] + 1].sum(axis=0) > 0)[0]
        cl = _trailing_cluster(cols)
        if cl and 140 <= cl[1] - cl[0] <= 340 and x_lo + cl[1] > 2150:
            rows = np.where(sbm[b[0]:b[1] + 1, cl[0]:cl[1] + 1].sum(axis=1) > 0)[0]
            sb_bbox = (x_lo + cl[0], b[0] + int(rows.min()),
                       x_lo + cl[1], b[0] + int(rows.max()))
            break
    if not sb_bbox: die(f'{note}: SHARED BY not found')
    dm = strict_dark(a[sb_bbox[3] + 2:sb_bbox[3] + 130, x_lo:])
    cols = np.where(dm.sum(axis=0) > 0)[0]
    cl = _trailing_cluster(cols, min_gap=70)
    if not cl or x_lo + cl[1] < 2150: die(f'{note}: name cluster not found')
    rows = np.where(dm[:, cl[0]:cl[1] + 1].sum(axis=1) > 0)[0]
    nm_bbox = (x_lo + cl[0], sb_bbox[3] + 2 + int(rows.min()),
               x_lo + cl[1], sb_bbox[3] + 2 + int(rows.max()))
    ph_bbox = None
    top = max(0, sb_bbox[1] - 380)
    pm = strict_dark(a[top:max(top + 1, sb_bbox[1] - 8), x_lo:])
    bb = blob_bbox(pm)
    if bb: ph_bbox = (x_lo + bb[0], top + bb[1], x_lo + bb[2], top + bb[3])
    if need_photo and not ph_bbox: die(f'{note}: photo blob not found')
    return sb_bbox, nm_bbox, ph_bbox

def normalize_onepager(path, gap_target=67, bottom_margin=90, out_path=None):
    """Uniform header-rule gap and bottom margin, matched to the gold layout.
    Everything from the rule down shifts as one block, so nothing can collide."""
    im = Image.open(path).convert('RGB')
    a = arr(im)
    H, W = a.shape[:2]
    dm = strict_dark(a[320:500, 1300:])
    cnt = dm.sum(axis=1)
    rows = np.where((cnt > 12) & (cnt < 600))[0]
    if not len(rows): die('normalize: name rows not found')
    name_bot = 320 + int(rows.max())
    dfull = strict_dark(a)
    rule_top = None
    for y in range(name_bot + 4, min(H, name_bot + 300)):
        if dfull[y].sum() > 1500:
            rule_top = y; break
    if rule_top is None: die('normalize: header rule not found')
    m = ink_mask(a, 225)
    cb = int(np.where(m.sum(axis=1) > 15)[0].max())
    target = name_bot + gap_target
    delta = target - rule_top
    new_H = cb + delta + bottom_margin
    canvas = Image.new('RGB', (W, new_H), (255, 255, 255))
    canvas.paste(im.crop((0, 0, W, rule_top - 2)), (0, 0))
    canvas.paste(im.crop((0, rule_top - 2, W, min(H, cb + 8))), (0, rule_top - 2 + delta))
    canvas.save(out_path or path)
    return dict(rule=(rule_top, target), height=(H, new_H))

# ------------------------------------------------------------ one-pager
def measure_onepager(gold_path):
    g = Image.open(gold_path); ga = arr(g)
    W, H = g.size
    m = {}
    ph = blob_bbox(ink_mask(ga[:520, 1850:], 235))
    if not ph: die('gold one-pager photo blob not found')
    m['photo'] = (1850 + ph[0], ph[1], 1850 + ph[2], ph[3])
    gm = green_mask(ga)
    bands = row_bands(gm, min_rows=20, min_px=60)
    cta = []
    for b in bands:
        cols = np.where(gm[b[0]:b[1] + 1].sum(axis=0) > 0)[0]
        if len(cols) and (cols.max() - cols.min()) > 0.7 * W:
            cta.append(b)
    if not cta: die(f'gold one-pager: no wide green band in {bands}')
    cta = sorted(cta, key=lambda b: b[0])
    b0, b1 = cta[-1]
    if not (0.6 * H < b0 and 50 <= b1 - b0 <= 150):
        die(f'gold one-pager bottom green band implausible: {(b0, b1)}')
    m['bar_band'] = (b0 - 8, b1 + 8)
    band_green = gm[b0:b1 + 1]
    cols = np.where(band_green.sum(axis=0) > 0)[0]
    left_cols = cols[cols < W * 0.72]
    code_cols = []
    for c in left_cols:
        colpx = band_green[:, c]
        if 6 < colpx.sum() < 0.8 * (b1 - b0):
            code_cols.append(c)
    if not code_cols: die('gold one-pager code columns not found')
    cx0, cx1 = min(code_cols), max(code_cols)
    rows = np.where(gm[b0:b1 + 1, cx0:cx1 + 1].sum(axis=1) > 0)[0]
    m['code'] = (cx0, b0 + rows.min(), cx1, b0 + rows.max())
    fm = color_mask(ga, (124, 139, 153), 36)
    fine = trailing_group(row_bands(fm[:, 100:2280], min_rows=4, min_px=40))
    if not fine or fine[0][0] < b1: die('gold one-pager fine print not found')
    m['fine_y0'] = fine[0][0]
    m['rule_y'] = last_rule_y(ga, m['bar_band'][0], 'gold one-pager')
    m['bar_off'] = m['bar_band'][0] - m['rule_y']
    m['fine_off'] = m['fine_y0'] - m['rule_y']
    hdr = dark_neutral(ga[:m['photo'][1] + 900, 1350:])
    m['size'] = (W, H)
    return m

def _name_bands(a, y_hi, x_lo=1350):
    dn = dark_neutral(a[:y_hi, x_lo:])
    bands = [(b[0], b[1]) for b in row_bands(dn, min_rows=8, min_px=25)]
    return bands, x_lo

def repair_onepager(broken_path, gold_path, partner, out_path, header='auto', norm_gap=67, norm_margin=90):
    m = measure_onepager(gold_path)
    W, H = m['size']
    gold = Image.open(gold_path); ga = arr(gold)
    src = Image.open(broken_path).convert('RGB')
    if src.size[0] != W: die(f'one-pager width mismatch {src.size}')
    im = Image.new('RGB', (W, H), (255, 255, 255))
    im.paste(src, (0, 0))
    a = arr(im)
    # 1. fine print down
    fm = color_mask(a, (124, 139, 153), 36)
    fine = trailing_group(row_bands(fm[:, 100:2280], min_rows=4, min_px=40))
    if not fine or fine[0][0] < H * 0.6: die(f'broken one-pager fine print implausible: {fine[:1]}')
    fb = fine[0]
    fe = fine[-1]
    ys, xs = np.where(fm[fb[0]:fe[1] + 1, :])
    fbox = (int(xs.min()), fb[0], int(xs.max()), fe[1])
    rule_b = last_rule_y(a, fb[0], 'broken one-pager')
    tgt_fine = rule_b + m['fine_off']
    dy = tgt_fine - fb[0]
    if dy < -170: die(f'one-pager fine print would move up ({dy})')
    if fe[1] + dy > H - 6: die('one-pager fine print would clip the canvas')
    if abs(dy) > 2:
        move_region(im, a, fbox, 0, dy, (255, 255, 255))
        a = arr(im)
    # 2. bar strip transplant with code re-render
    gb0, gb1 = m['bar_band']
    bb0 = rule_b + m['bar_off']
    bb1 = bb0 + (gb1 - gb0)
    wipe_stray_rules(im, a, rule_b + 8, tgt_fine - 12)
    a = arr(im)
    zone = a[bb0:bb1 + 1]
    if ink_mask(zone, 245).sum() > 40: die('one-pager bar zone not clear after move')
    strip = gold.crop((0, gb0, W, gb1 + 1)).convert('RGB')
    cx0, cy0, cx1, cy1 = m['code']
    sa = arr(strip)
    bg = sample_bg(sa, cx0 - 14, cy0 - gb0 + (cy1 - cy0) // 2)
    gsamp = sa[(cy0 - gb0):(cy1 - gb0) + 1, cx0:cx1 + 1]
    gm2 = green_mask(gsamp)
    gcol = tuple(int(v) for v in gsamp[gm2].mean(axis=0)) if gm2.sum() else (14, 122, 95)
    ImageDraw.Draw(strip).rectangle((cx0 - 2, cy0 - gb0 - 2, cx1 + 3, cy1 - gb0 + 3), fill=bg)
    ch = cy1 - cy0 + 1
    fs = int(ch * 1.35)
    while fs > 8:
        f = ImageFont.truetype(JBMB, fs)
        bb = f.getbbox(partner['code'])
        if bb[3] - bb[1] <= ch + 2: break
        fs -= 1
    f = ImageFont.truetype(JBMB, fs)
    bb = f.getbbox(partner['code'])
    ImageDraw.Draw(strip).text((cx0 - bb[0], cy0 - gb0 - bb[1]), partner['code'],
                               font=f, fill=gcol)
    im.paste(strip, (0, bb0))
    a = arr(im)
    # 3+4. header: photo slot state decides the strategy
    if header == 'skip':
        im.save(out_path)
        normalize_onepager(out_path, norm_gap, norm_margin)
        return dict(bar_band=(bb0, bb1), photo=None, fine_dy=dy)
    g_sb, g_nm, g_ph = GOLD_OP_SB, GOLD_OP_NAME, GOLD_OP_PHOTO
    side = max(g_ph[2] - g_ph[0], g_ph[3] - g_ph[1])
    ph_sq = (g_ph[0], g_ph[1], g_ph[0] + side, g_ph[1] + side)
    tail = partner['full'] != partner['clean']
    n_words = len(partner['full'].split())
    st = photo_state(a, ph_sq, partner['photo'])
    if st in ('disc', 'match'):
        # header never reflowed; fix name in place, cover disc if present
        if tail:
            dm = strict_dark(a)
            bands = [b for b in row_bands(dm[:, 1300:], min_rows=10, min_px=20)
                     if b[0] < g_nm[3] + 40 and b[1] > g_nm[1] - 40]
            if len(bands) != 1: die(f'one-pager in-place name band ambiguous: {bands}')
            nm_band = bands[0]
            strip_tail_words(im, a, nm_band, 1300, W - 20, n_words, 2,
                             'one-pager name', mask_fn=strict_dark)
            a = arr(im)
            dm = strict_dark(a)
            cols = np.where(dm[nm_band[0]:nm_band[1] + 1, 1300:].sum(axis=0) > 0)[0]
            if len(cols) == 0: die('one-pager name vanished after strip')
            nm_box = (1300 + int(cols.min()), nm_band[0], 1300 + int(cols.max()), nm_band[1])
            move_region(im, a, nm_box, g_nm[2] - nm_box[2], 0, (255, 255, 255))
            a = arr(im)
        if st == 'disc':
            paste_photo(im, ph_sq, partner['photo'], cover_disc=True)
    elif st == 'empty':
        b_sb, b_nm, b_ph = header_block(a, 'broken one-pager')
        if tail:
            keep_x1, _ = strip_tail_words(im, a, (b_nm[1], b_nm[3]), b_nm[0] - 4,
                                          b_nm[2] + 6, n_words, 2, 'one-pager name',
                                          mask_fn=strict_dark)
            a = arr(im)
            b_nm = (b_nm[0], b_nm[1], keep_x1, b_nm[3])
        if abs((b_nm[3] - b_nm[1]) - (g_nm[3] - g_nm[1])) > 14:
            die(f'one-pager name height mismatch vs gold: {b_nm} vs {g_nm}')
        move_region(im, a, b_sb, g_sb[2] - b_sb[2], g_sb[1] - b_sb[1], (255, 255, 255))
        a = arr(im)
        move_region(im, a, b_nm, g_nm[2] - b_nm[2], g_nm[1] - b_nm[1], (255, 255, 255))
        a = arr(im)
        paste_photo(im, ph_sq, partner['photo'])
    else:
        die('one-pager photo region state: ' + st)
    im.save(out_path)
    return dict(bar_band=(bb0, bb1), photo=ph_sq, fine_dy=dy)

# ------------------------------------------------------- banner and cover
def repair_flat_asset(broken_path, gold_path, partner, out_path, kind, photo_action='fix'):
    gold = Image.open(gold_path); ga = arr(gold)
    im = Image.open(broken_path).convert('RGB')
    a = arr(im)
    if im.size != gold.size: die(f'{kind} size mismatch {im.size} vs {gold.size}')
    W, H = im.size
    if kind == 'banner':
        gb = blob_bbox(ink_mask(ga[:, :640], 235))
    else:
        gb = blob_bbox(ink_mask(ga[:1000, 1400:], 235))
        gb = (1400 + gb[0], gb[1], 1400 + gb[2], gb[3]) if gb else None
    if not gb: die(f'gold {kind} photo blob not found')
    if partner['full'] != partner['clean']:
        n = len(partner['full'].split())
        if kind == 'banner':
            reg_x0 = gb[2] + 60
            dn = dark_neutral(a[:, reg_x0:])
            bands = row_bands(dn, min_rows=30, min_px=40)
            if not bands: die('banner name band not found')
            nm = max(bands, key=lambda b: b[1] - b[0])
            strip_tail_words(im, a, nm, reg_x0, min(W - 10, 2500), n, 2, 'banner name')
        else:
            dn = dark_neutral(a[150:1100, 80:1300])
            bands = [(b[0] + 150, b[1] + 150) for b in row_bands(dn, min_rows=15, min_px=25)]
            tgt = None
            for b in bands:
                g = col_groups(dark_neutral(a), b, 80, 2140, gap_frac=0.27)
                if len(g) == n + 4:       # 'Five resources, shared by' + name words
                    tgt = (b, g); break
            if not tgt: die(f'cover name line not found among {len(bands)} bands')
            b, g = tgt
            last = g[-1]                   # 'Test.'
            sub = dark_neutral(a[b[0]:b[1] + 1, last[0]:last[1] + 1])
            colp = np.where(sub.sum(axis=0) > 0)[0]
            gaps = np.where(np.diff(colp) > 3)[0]
            if not len(gaps): die('cover: period not separable from Test.')
            dot_x0 = last[0] + int(colp[gaps[-1] + 1])
            dot = im.crop((dot_x0 - 1, b[0] - 2, last[1] + 3, b[1] + 3))
            grad_fill(im, a, g[-2][0] - 4, b[0] - 4, last[1] + 5, b[1] + 5)
            im.paste(dot, (g[-3][1] + 3, b[0] - 2))
    a = arr(im)
    if photo_action == 'fix':
        st = photo_state(a, gb, partner['photo'])
        if st in ('disc', 'empty'):
            paste_photo(im, gb, partner['photo'], cover_disc=(st == 'disc'))
        elif st != 'match':
            die(f'{kind} photo region state: {st}')
    im.save(out_path)
