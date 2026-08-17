"""Prospeo CTA repair, v2 individual-pack path.
Philosophy per handoff: surgical stamps only, measure the broken doc, fail loud.
Gold geometry: Chris Ritson v2 pack (all four guides share identical p3 geometry)."""
import fitz, os, io, numpy as np
from PIL import Image, ImageDraw

JBMB = '/home/claude/fonts/JetBrainsMono-Bold.ttf'
GREEN = (0.055, 0.478, 0.373)          # #0e7a5f
WHITE = (1, 1, 1)
LGREY = (0.663, 0.729, 0.784)          # #a9bac8
LINKC = (0.561, 0.827, 0.753)          # #8fd3c0
GOLD_CODE = 'CHRISR50'
GOLD_NAME = 'Chris Ritson'

# ---- v2 calibration (measured from Chris gold; guarded at runtime) ----
G3 = dict(                              # guide p3 (identical across cc/icp/six/obj)
    bar_rect=(30.0, 720.8, 564.8, 758.2),
    bar_clip=(29.0, 719.8, 565.8, 759.4),   # incl. border + divider below
    code_x=185.8, code_base=743.2, code_size=11.3,
    photo_rect=(513.8, 27.0, 564.8, 78.8),
    sb_rect=(512.6, 84.2, 564.0, 91.6), sb_base=91.5,
    name_right=565.0, name_base=105.8,
    foot_first_base=777.8, foot_x0=29.7,
)
RP1 = dict(                             # report p1
    offer_base=186.9, offer_x0=36.7, offer_size=6.3,
    divider=(36.7, 200.4, 559.4, 200.9), divider_fill=(0.16, 0.21, 0.25),
    label_base=219.0, value_base=250.6, dark_edge=281.8,
    photo_rect=None,                    # measured from gold at runtime
    sb_base=149.7, sb_x=67.4, name_base=162.3, name_x=67.4,
)
RP2 = dict(banner_clip=None, fine_first_base=406.4, code_base=377.6, code_x=252.6,
           code_size=7.0, link_rect=(490.6, 365.5, 548.7, 385.5))

import re
_GOLD_CODE_CACHE = {}
def gold_code(gold_doc):
    key = id(gold_doc)
    if key not in _GOLD_CODE_CACHE:
        m = []
        for i in range(gold_doc.page_count):
            for l in gold_doc[i].get_links():
                m = re.findall(r'coupon=([A-Z]+50)', l.get('uri', ''))
                if m: break
            if m: break
        if not m: raise SystemExit('GUARD FAILED: gold code not found in gold links')
        _GOLD_CODE_CACHE[key] = m[0]
    return _GOLD_CODE_CACHE[key]

def die(msg):
    raise SystemExit('GUARD FAILED: ' + msg)

def spans(page, clip=None):
    out = []
    for b in page.get_text('dict', clip=clip)['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                out.append(s)
    return out

def find_span(page, text, ymin=0, ymax=9999):
    for s in spans(page):
        if text in s['text'] and ymin <= s['origin'][1] <= ymax:
            return s
    return None

def char_rects(page, text):
    """Exact per-char rects for the first occurrence of text on the page."""
    raw = page.get_text('rawdict')
    chars = []
    for b in raw['blocks']:
        for l in b.get('lines', []):
            for s in l.get('spans', []):
                for c in s.get('chars', []):
                    chars.append((c['c'], fitz.Rect(c['bbox'])))
    flat = ''.join(c for c, _ in chars)
    i = flat.find(text)
    if i < 0:
        return None
    return [r for _, r in chars[i:i + len(text)]]

def redact(page, rect, fill=None):
    page.add_redact_annot(fitz.Rect(rect), fill=fill if fill else False)

def apply_red(page):
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_NONE)

def circular_png(photo_path, px=512):
    im = Image.open(photo_path).convert('RGB').resize((px, px), Image.LANCZOS)
    m = Image.new('L', (px * 4, px * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, px * 4 - 1, px * 4 - 1), fill=255)
    im.putalpha(m.resize((px, px), Image.LANCZOS))
    buf = io.BytesIO(); im.save(buf, 'PNG'); return buf.getvalue()

def stamp_code(page, code, x, base, size):
    page.insert_text((x, base), code, fontsize=size, fontname='JBMB',
                     fontfile=JBMB, color=GREEN)

# ---------------------------------------------------------------- guide p3
def repair_guide_p3(doc, pristine, gold_cc, partner, notes):
    """Photo, header name block, CTA bar + link, footnote position. Page index 2."""
    page, ppage, gpage = doc[2], pristine[2], gold_cc[2]
    full, clean, code = partner['full'], partner['clean'], partner['code']

    # -- 1. photo --------------------------------------------------------
    slot = [im for im in page.get_images(full=True) if im[2] == 512 and im[3] == 512]
    if partner['photo_mode'] == 'replace':
        if not slot: die('replace mode but no 512 slot on ' + notes)
        doc[2].replace_image(slot[0][0], filename=partner['photo'])
    elif partner['photo_mode'] == 'insert':
        if slot: die('insert mode but slot exists on ' + notes)
        page.insert_image(fitz.Rect(G3['photo_rect']),
                          stream=circular_png(partner['photo']))
    elif partner['photo_mode'] == 'keep':
        if not slot: die('keep mode but no photo present on ' + notes)

    # -- 2. header: SHARED BY + name ------------------------------------
    sb = find_span(page, 'SHARED BY', 0, 200)
    nm_first = clean.split()[0]
    nm = find_span(page, nm_first, 0, 200)
    if not sb or not nm: die('header spans missing on ' + notes)
    sb_r, nm_r = fitz.Rect(sb['bbox']), fitz.Rect(nm['bbox'])
    moved = abs(nm['origin'][1] - G3['name_base']) > 0.6
    tail = full != clean
    if moved or tail:
        # pristine copy: strip tail chars if needed, then measure clean ink
        if tail:
            cr = char_rects(ppage, full)
            if not cr or len(cr) != len(full): die('char map failed on ' + notes)
            sp = cr[len(clean)]
            tail_rect = fitz.Rect(sp.x0 + 0.25 * sp.width, sp.y0 - 0.5,
                                  cr[-1].x1 + 1, cr[-1].y1 + 0.5)
            redact(ppage, tail_rect); apply_red(ppage)
            if 'Lead Test' in ppage.get_text(): die('tail strip incomplete on ' + notes)
        src = ppage.search_for(clean, clip=fitz.Rect(0, 0, 600, 200))
        if not src: die('clean name not found on pristine ' + notes)
        src_r = src[0]
        p_span = find_span(ppage, nm_first, 0, 200)
        base_off = p_span['origin'][1] - src_r.y0
        # redact originals on target (name line + SHARED BY if it moved)
        redact(page, nm_r + (-1, -1, 1, 1))
        if moved: redact(page, sb_r + (-1, -1, 1, 1))
        apply_red(page)
        ty0 = G3['name_base'] - base_off
        page.show_pdf_page(fitz.Rect(G3['name_right'] - src_r.width, ty0,
                                     G3['name_right'], ty0 + src_r.height),
                           pristine, 2, clip=src_r)
        if moved:
            psb = ppage.search_for('SHARED BY', clip=fitz.Rect(0, 0, 600, 200))
            if not psb: die('SHARED BY missing on pristine ' + notes)
            sr = psb[0]
            page.show_pdf_page(fitz.Rect(564.0 - sr.width, G3['sb_rect'][1],
                                         564.0, G3['sb_rect'][1] + sr.height),
                               pristine, 2, clip=sr)

    # -- 3+4. CTA zone, positioned relative to the metrics row -----------
    # every offset is measured from the gold pack itself at runtime
    gpage0 = gold_cc[2]
    gml0 = find_span(gpage0, 'Metrics led', 600, 800)
    if not gml0: die('gold Metrics led anchor missing')
    gmb0 = gml0['origin'][1]
    g_bars = [d['rect'] for d in gpage0.get_drawings()
              if d.get('fill') and d['rect'].width > 480 and 30 < d['rect'].height < 50
              and abs(d['fill'][0] - 0.965) < 0.02]
    if len(g_bars) != 1: die(f'gold p3 bar rect ambiguous: {len(g_bars)}')
    gbr = g_bars[0]
    g_code_sp = find_span(gpage0, gold_code(gold_cc), gbr.y0, gbr.y1)
    g_foot = [s for s in spans(gpage0) if s['origin'][1] > gbr.y1 + 5
              and 'Georgia' in s['font'] and s['size'] < 9]
    if not g_code_sp or not g_foot: die('gold p3 code/footnote not found')
    off_bar0, off_bar1 = gbr.y0 - gmb0, gbr.y1 - gmb0
    off_code_x = g_code_sp['origin'][0]
    off_code_b = g_code_sp['origin'][1] - gmb0
    off_foot = min(s['origin'][1] for s in g_foot) - gmb0
    g_foot_y1 = max(fitz.Rect(s['bbox']).y1 for s in g_foot)
    ml = find_span(page, 'Metrics led', 600, 800)
    if not ml: die('Metrics led anchor missing on ' + notes)
    mb = ml['origin'][1]
    bar_rect = (30.0, mb + off_bar0, 564.8, mb + off_bar1)
    bar_clip_t = (29.0, mb + off_bar0 - 1.0, 565.8, mb + off_bar1 + 1.2)
    foot_target = mb + off_foot
    # footnote reposition (partner content; may move a few pt either way)
    foot = [s for s in spans(page) if s['origin'][1] > mb + 30 and 'Georgia' in s['font']
            and s['size'] < 9]
    if not foot: die('footnote not found on ' + notes)
    fb = min(s['origin'][1] for s in foot)
    shift = foot_target - fb
    if shift < -40: die(f'footnote would move up {-shift:.1f}pt on ' + notes)
    x0 = min(fitz.Rect(s['bbox']).x0 for s in foot)
    y0 = min(fitz.Rect(s['bbox']).y0 for s in foot)
    x1 = max(fitz.Rect(s['bbox']).x1 for s in foot)
    y1 = max(fitz.Rect(s['bbox']).y1 for s in foot)
    max_y1 = max(page.rect.height - 0.2, g_foot_y1 + 0.5)
    if y1 + shift > max_y1: die('footnote would clip page on ' + notes)
    if abs(shift) > 0.4:
        redact(page, (x0 - 1, y0 - 1, x1 + 1, y1 + 1)); apply_red(page)
        page.show_pdf_page(fitz.Rect(x0, y0 + shift, x1, y1 + shift),
                           pristine, 2, clip=fitz.Rect(x0, y0, x1, y1))
    # clear any orphaned line art between the bar zone and the footnote
    clear_zone = fitz.Rect(29.0, mb + 18.6, 565.8, foot_target - 11)
    own = page.get_text(clip=clear_zone).strip()
    if own: die(f'target bar zone not clear on {notes}: "{own[:60]}"')
    redact(page, clear_zone)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    # transplant the gold bar block (template + code only)
    gpage = gold_cc[2]
    gold_clip = (29.0, gbr.y0 - 1.0, 565.8, gbr.y1 + 1.2)
    bar_txt = ' '.join(gpage.get_text(clip=fitz.Rect(gold_clip)).split())
    gc_ = gold_code(gold_cc)
    if gc_ not in bar_txt or len(bar_txt) > 80:
        die('gold bar clip contains unexpected content: ' + bar_txt)
    gcopy = fitz.open()
    gcopy.insert_pdf(gold_cc, from_page=2, to_page=2)
    gc = gcopy[0]
    for r in gc.search_for(gold_code(gold_cc)):
        redact(gc, r)
    apply_red(gc)
    page.show_pdf_page(fitz.Rect(bar_clip_t), gcopy, 0, clip=fitz.Rect(gold_clip))
    gcopy.close()
    stamp_code(page, code, off_code_x, mb + off_code_b, G3['code_size'])
    page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(bar_rect),
                      'uri': partner['uri']})

# ------------------------------------------------------------ closing pages
FOOTER_BAND = (36.0, 825.9, 559.0, 839.9)
FOOTER_BASE = 834.5
CLOSING_RECTS = [(337.3, 503.2, 400.7, 539.0), (337.3, 536.8, 401.4, 546.2)]

def _closing_anchor_spans(page, notes):
    big = [s for s in spans(page) if s['text'].strip() == '50%' and s['size'] > 25]
    off = [s for s in spans(page) if s['text'].strip().startswith('off your first year')]
    if len(big) != 1 or len(off) != 1:
        die(f'closing anchors not unique on {notes}: {len(big)}/{len(off)}')
    return fitz.Rect(big[0]['bbox']), fitz.Rect(off[0]['bbox'])

def repair_closing(doc, gold_cc, pno, partner, notes):
    """Restore 2 closing-block links + footer line on cc p13 / retainer p14.
    All geometry and footer text derived from the gold's own closing page."""
    page, gpage = doc[pno], gold_cc[12]
    g_big, g_off = _closing_anchor_spans(gpage, 'gold cc p13')
    t_big, t_off = _closing_anchor_spans(page, notes)
    grects = sorted([fitz.Rect(l['from']) for l in gpage.get_links()
                     if l.get('kind') == fitz.LINK_URI and fitz.Rect(l['from']).y1 < 800],
                    key=lambda r: r.y0)
    if len(grects) != 2: die(f'gold closing links != 2: {len(grects)}')
    for gold_rect, g_a, t_a in [(tuple(grects[0]), g_big, t_big),
                                (tuple(grects[1]), g_off, t_off)]:
        gr = fitz.Rect(gold_rect)
        r = fitz.Rect(t_a.x0 + (gr.x0 - g_a.x0), t_a.y0 + (gr.y0 - g_a.y0),
                      t_a.x1 + (gr.x1 - g_a.x1), t_a.y1 + (gr.y1 - g_a.y1))
        page.insert_link({'kind': fitz.LINK_URI, 'from': r, 'uri': partner['uri']})
    if page.get_text(clip=fitz.Rect(36, 822, 559, 843)).strip():
        die('footer zone not clear on ' + notes)
    g_ft = ' '.join(gpage.get_text(clip=fitz.Rect(30, 822, 566, 843)).split())
    if not g_ft: die('gold footer text not found')
    txt = g_ft.replace(gold_code(gold_cc), partner['code'])
    helv = fitz.Font('helv')
    w = helv.text_length(txt, 8.0)
    page.insert_text(((page.rect.width - w) / 2, FOOTER_BASE), txt,
                     fontsize=8.0, fontname='helv', color=(0.322, 0.361, 0.42))
    page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(FOOTER_BAND),
                      'uri': partner['uri']})

FOOTER_TXT = ('50% off Prospeo for readers of this guide - code CHRISR50 - '
              'prospeo.io/pricing?coupon=CHRISR50')

# ------------------------------------------------------------------ report
def _p1_solver(page, notes):
    """Equal margins for offer line + divider in the name->statband gap."""
    ss = spans(page)
    cand = [s for s in ss if s['origin'][1] < 220 and 7 <= s['size'] <= 12 and
            s['color'] == 0xFFFFFF and len(s['text'].strip()) > 5]
    name = sorted(cand, key=lambda s: s['origin'][1])[-1:]
    lab = [s for s in ss if 'LEADS TESTED' in s['text']]
    if not name or not lab: die('p1 anchors missing on ' + notes)
    name_base = name[0]['origin'][1]
    lab_cap = lab[0]['origin'][1] - 4.4
    # the stat band's own top border rule bounds the zone, not the label text
    rules = [d['rect'] for d in page.get_drawings()
             if d.get('fill') and d['rect'].width > 450 and d['rect'].height < 1.6
             and name_base + 2 < d['rect'].y0 < lab_cap + 6]
    zone_end = min(r.y0 for r in rules) if rules else lab_cap
    LINE_H = 5.8
    m = (zone_end - name_base - LINE_H) / 2
    if m < 4.0: die(f'p1 margin {m:.1f} below floor on ' + notes)
    line_base = name_base + m + LINE_H
    if line_base + 2.0 > zone_end - 2.0:
        die(f'p1 offer line would touch the band rule on ' + notes)
    return line_base, m

def repair_report(doc, pristine, gold_rp, partner, notes):
    p1, gp1 = doc[0], gold_rp[0]
    full, clean, code = partner['full'], partner['clean'], partner['code']
    # photo
    slots = [im for im in p1.get_images(full=True) if im[2] == 512 and im[3] == 512]
    if partner['photo_mode'] == 'replace':
        if not slots: die('report replace mode but no slot ' + notes)
        doc[0].replace_image(slots[0][0], filename=partner['photo'])
    elif partner['photo_mode'] == 'insert':
        if slots: die('report insert mode but slot exists ' + notes)
        p1.insert_image(fitz.Rect(36.7, 143.7, 60.9, 167.9),
                        stream=circular_png(partner['photo']))
    elif partner['photo_mode'] == 'keep':
        if not slots: die('report keep mode but no photo ' + notes)
    # header: name x reposition (photo drop slides block left) + tail strip
    pp1 = pristine[0]
    nm = find_span(p1, clean.split()[0], 100, 200)
    sb = find_span(p1, 'SHARED BY', 100, 200)
    if not nm or not sb: die('p1 header spans missing ' + notes)
    slid = abs(nm['origin'][0] - 67.4) > 0.8
    tail = full != clean
    if tail:
        cr = char_rects(pp1, full)
        if not cr or len(cr) != len(full): die('p1 char map failed ' + notes)
        sp = cr[len(clean)]
        redact(pp1, fitz.Rect(sp.x0 + 0.25 * sp.width, sp.y0 - 0.5,
                              cr[-1].x1 + 1, cr[-1].y1 + 0.5))
        apply_red(pp1)
        if 'Lead Test' in pp1.get_text(): die('p1 tail strip incomplete ' + notes)
    if slid or tail:
        src_n = pp1.search_for(clean, clip=fitz.Rect(30, 130, 580, 200))
        src_s = pp1.search_for('SHARED BY', clip=fitz.Rect(30, 130, 580, 200))
        if not src_n or not src_s: die('p1 pristine header not found ' + notes)
        rn, rs = src_n[0], src_s[0]
        pn = find_span(pp1, clean.split()[0], 100, 200)
        ps = find_span(pp1, 'SHARED BY', 100, 200)
        redact(p1, fitz.Rect(nm['bbox']) + (-1, -1, 1, 1))
        redact(p1, fitz.Rect(sb['bbox']) + (-1, -1, 1, 1))
        apply_red(p1)
        ny0 = 162.3 - (pn['origin'][1] - rn.y0)
        p1.show_pdf_page(fitz.Rect(67.4, ny0, 67.4 + rn.width, ny0 + rn.height),
                         pristine, 0, clip=rn)
        sy0 = 149.7 - (ps['origin'][1] - rs.y0)
        p1.show_pdf_page(fitz.Rect(67.4, sy0, 67.4 + rs.width, sy0 + rs.height),
                         pristine, 0, clip=rs)
    # ---- p1: proportional header solve (gaps scaled to gold rhythm) ----
    def _p1_measure(pg):
        ss_ = spans(pg)
        cand = [s for s in ss_ if s['origin'][1] < 230 and 7 <= s['size'] <= 12 and
                s['color'] == 0xFFFFFF and len(s['text'].strip()) > 5]
        nb = sorted(cand, key=lambda s: s['origin'][1])[-1]['origin'][1]
        lab = [s for s in ss_ if 'LEADS TESTED' in s['text']]
        if not lab: die('p1 LEADS TESTED missing ' + notes)
        lb = lab[0]['origin'][1]
        vals = [s for s in ss_ if s['size'] > 9.5 and s['color'] == 0xFFFFFF
                and lb < s['origin'][1] < lb + 60]
        if not vals: die('p1 stat values missing ' + notes)
        vbot = max(fitz.Rect(s['bbox']).y1 for s in vals)
        rules = [d['rect'] for d in pg.get_drawings() if d.get('fill')
                 and d['rect'].width > 450 and d['rect'].height < 1.6
                 and nb + 2 < d['rect'].y0 < lb]
        if not rules: die('p1 band rule missing ' + notes)
        rt = min(r.y0 for r in rules)
        dark = [d['rect'] for d in pg.get_drawings() if d.get('fill')
                and abs(d['fill'][0] - 0.06) < 0.02 and d['rect'].width > 500]
        if not dark: die('p1 dark header missing ' + notes)
        edge = max(d.y1 for d in dark)
        return nb, rt, vbot, edge
    g_off = find_span(gold_rp[0], 'Readers get', 150, 230)
    if not g_off: die('gold p1 offer anchor missing')
    g_nb, g_rt, g_vb, g_edge = _p1_measure(gold_rp[0])
    G1 = (g_off['origin'][1] - 5.8) - g_nb
    G2 = g_rt - g_off['origin'][1]
    G3 = g_edge - g_vb
    nb, rt, vb, edge = _p1_measure(p1)
    band_h = vb - rt
    avail = edge - nb
    scale = (avail - 5.8 - band_h) / (G1 + G2 + G3)
    if scale < 0.45: die(f'p1 header too compressed (scale {scale:.2f}) on ' + notes)
    m1 = round(min(G1, G2) * scale, 1)
    line_base = nb + G1 * scale + 5.8
    new_rt = line_base + G2 * scale
    delta = new_rt - rt
    if abs(delta) > 1.5:
        band_rect = fitz.Rect(29.0, rt - 1.2, 566.0, vb + 2.8)
        p1.add_redact_annot(band_rect, fill=(0.06, 0.09, 0.13))
        p1.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
        p1.show_pdf_page(band_rect + (0, delta, 0, delta), pristine, 0, clip=band_rect)
    guard = p1.get_text(clip=fitz.Rect(36, line_base - 8, 560, line_base + 3)).strip()
    if guard: die(f'p1 offer zone not clear on {notes}: "{guard[:40]}"')
    helv, hebo, jbm = fitz.Font('helv'), fitz.Font('hebo'), fitz.Font(fontfile=JBMB)
    x = 36.7
    segs = [('Readers get ', helv, 'helv', None, LGREY),
            ('50% off Prospeo', hebo, 'hebo', None, WHITE),
            (' with code ', helv, 'helv', None, LGREY),
            (code, jbm, 'JBMB', JBMB, WHITE),
            ('  \u00b7  ', helv, 'helv', None, LGREY),
            ('https://prospeo.io/pricing?coupon=' + code, helv, 'helv', None, LINKC)]
    for txt, fo, fn, ff, col in segs:
        p1.insert_text((x, line_base), txt, fontsize=6.3, fontname=fn,
                       fontfile=ff, color=col)
        if col == LINKC:
            w_ = fo.text_length(txt, 6.3)
            p1.insert_link({'kind': fitz.LINK_URI, 'uri': partner['uri'],
                            'from': fitz.Rect(x, line_base - 6, x + w_, line_base + 1.5)})
            p1.draw_rect(fitz.Rect(x, line_base + 1.0, x + w_, line_base + 1.5),
                         color=None, fill=LINKC)
        x += fo.text_length(txt, 6.3)
    # ---- p2 banner ----
    p2, gp2 = doc[1], gold_rp[1]
    ss2 = spans(p2)
    ml = [s for s in ss2 if 'Metrics led' in s['text']]
    meth = [s for s in ss2 if 'M E T H' in s['text']]
    if not ml or not meth: die('p2 anchors missing ' + notes)
    # chart bottom: lowest ranking-row span above methodology
    chart_low = max(s['origin'][1] for s in ss2
                    if s['origin'][1] < meth[0]['origin'][1] - 60 and s['size'] > 6.4)
    meth_cap = meth[0]['origin'][1] - 4.5
    fine = sorted([s for s in ss2 if 'Georgia' in s['font'] and s['size'] < 6.4
                   and chart_low + 2 < s['origin'][1] < meth_cap],
                  key=lambda s: s['origin'][1])
    g_bans0 = [d['rect'] for d in gp2.get_drawings()
               if d.get('fill') and d['rect'].width > 480 and 30 < d['rect'].height < 50
               and abs(d['fill'][0] - 0.965) < 0.02]
    BAN_H = g_bans0[0].height if g_bans0 else 394.3 - 356.7
    if fine:
        f_rs = [fitz.Rect(s['bbox']) for s in fine]
        fx0 = min(r.x0 for r in f_rs); fy0 = min(r.y0 for r in f_rs)
        fx1 = max(r.x1 for r in f_rs); fy1 = max(r.y1 for r in f_rs)
        fine_h = fy1 - fy0
        zone = meth_cap - (chart_low + 2)
        m2 = (zone - BAN_H - fine_h) / 3
        if m2 < 4.5: die(f'p2 margin {m2:.1f} impossible on ' + notes)
        ban_top = chart_low + 2 + m2
        redact(p2, (fx0 - 1, fy0 - 1, fx1 + 1, fy1 + 1)); apply_red(p2)
        p2.show_pdf_page(fitz.Rect(fx0, ban_top + BAN_H + m2, fx1,
                                   ban_top + BAN_H + m2 + fine_h),
                         pristine, 1, clip=fitz.Rect(fx0, fy0, fx1, fy1))
    else:
        zone = meth_cap - (chart_low + 2)
        m2 = (zone - BAN_H) / 2
        if m2 < 7.5: die(f'p2 margin {m2:.1f} below floor on ' + notes)
        ban_top = chart_low + 2 + m2
    guard2 = p2.get_text(clip=fitz.Rect(36, ban_top - 1, 560, ban_top + BAN_H + 1)).strip()
    if guard2: die(f'p2 banner zone not clear on {notes}: "{guard2[:40]}"')
    g_bans = [d['rect'] for d in gp2.get_drawings()
              if d.get('fill') and d['rect'].width > 480 and 30 < d['rect'].height < 50
              and abs(d['fill'][0] - 0.965) < 0.02]
    if len(g_bans) != 1: die(f'gold p2 banner rect ambiguous: {len(g_bans)}')
    gb = g_bans[0]
    g_clip = fitz.Rect(gb.x0 - 0.5, gb.y0 - 0.5, gb.x1 + 0.6, gb.y1 + 0.6)
    gcode = gold_code(gold_rp)
    gtxt = ' '.join(gp2.get_text(clip=g_clip).split())
    if gcode not in gtxt: die('gold banner clip bad: ' + gtxt[:60])
    g_cd = gp2.search_for(gcode, clip=g_clip)
    g_lk = [fitz.Rect(l['from']) for l in gp2.get_links()
            if l.get('kind') == fitz.LINK_URI and fitz.Rect(l['from']).intersects(g_clip)]
    if not g_cd or not g_lk: die('gold banner code/link not found')
    gcopy = fitz.open(); gcopy.insert_pdf(gold_rp, from_page=1, to_page=1)
    gc = gcopy[0]
    for r in gc.search_for(gcode, clip=g_clip):
        redact(gc, r)
    apply_red(gc)
    dy = ban_top - gb.y0
    p2.show_pdf_page(g_clip + (0, dy, 0, dy), gcopy, 0, clip=g_clip)
    gcopy.close()
    # partner code at the gold code position (baseline approximated from span box)
    stamp_code(p2, code, g_cd[0].x0, g_cd[0].y1 - 1.5 + dy, RP2['code_size'])
    p2.insert_link({'kind': fitz.LINK_URI, 'uri': partner['uri'],
                    'from': g_lk[0] + (0, dy, 0, dy)})
    return m1, m2
