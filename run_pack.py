#!/usr/bin/env python3
"""One-command partner pack repair. Everything (unpack, repair, audit, proof) in one run.

Usage:
  python3 run_pack.py --zip "/path/broken.zip" --name "Erik Foreman" --code ERIKF50 \
      --audience agency [--headshot /path/photo.jpg] [--gold-zip /path/leader.zip]

Outputs to /mnt/user-data/outputs: repaired pack zip + proof sheet. Prints PASS or FAIL.
"""
import argparse, os, re, shutil, sys, zipfile, tempfile
import fitz, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repair_v2 as R
import repair_png as P

KNOWN_GUIDES = ["Cold Call Openers", "ICP Targeting Guide", "Six-Email Sequence", "Retainer Structures"]
CLOSING = {"Cold Call Openers", "Retainer Structures"}
GOLD_NAME_REMNANTS = ["Lead Test", "Chris Ritson", "Jed Mahrle", "CHRISR50", "JEDM50"]
OUT = "/mnt/user-data/outputs"

def fail(msg):
    print("FAIL: " + msg); sys.exit(1)

def unzip_to(zp, d):
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(zp) as z: z.extractall(d)
    # flatten single-folder zips
    entries = os.listdir(d)
    if len(entries) == 1 and os.path.isdir(os.path.join(d, entries[0])):
        inner = os.path.join(d, entries[0])
        for f in os.listdir(inner): shutil.move(os.path.join(inner, f), d)
        os.rmdir(inner)
    return d

def manifest(d):
    files = [f for f in os.listdir(d) if ' - ' in f]
    if not files: fail("no pack files found in the zip (expected 'Name - Document' naming)")
    prefixes = {f.rsplit(' - ', 1)[0] for f in files}
    if len(prefixes) != 1: fail(f"mixed file prefixes in zip: {sorted(prefixes)}")
    full = prefixes.pop()
    docs = {f.rsplit(' - ', 1)[1].rsplit('.', 1)[0]: os.path.join(d, f) for f in files}
    return full, docs

def gold_paths(repo, audience, leader_dir):
    pre = {"sales": "individual", "agency": "group"}[audience]
    g = {"cc": os.path.join(repo, f"{pre}-gold-cold-call-openers.pdf"),
         "report": os.path.join(repo, f"{pre}-gold-report.pdf"),
         "onepager": os.path.join(repo, f"{pre}-gold-onepager.png"),
         "banner": os.path.join(repo, f"{pre}-gold-banner.png"),
         "cover": os.path.join(repo, f"{pre}-gold-cover.png")}
    if leader_dir:
        _, ldocs = manifest(leader_dir)
        m = {"cc": "Cold Call Openers", "report": "Benchmark Report",
             "onepager": "Benchmark One-Pager", "banner": "Banner", "cover": "Cover"}
        for k, doc in m.items():
            if doc in ldocs: g[k] = ldocs[doc]
    for k, p_ in g.items():
        if not os.path.exists(p_): fail(f"gold file missing: {p_}")
    return g

def photo_plan(docs, headshot):
    cc = fitz.open(docs["Cold Call Openers"])
    slot = [im for im in cc[2].get_images(full=True) if im[2] == 512 and im[3] == 512]
    cc.close()
    if slot and headshot: return "replace"
    if slot: return "keep"
    if headshot: return "insert"
    fail("photos are missing from the pack and no headshot was attached; attach the partner headshot and rerun")

def gold_op_metrics(gop):
    a = P.arr(Image.open(gop)); H, W = a.shape[:2]
    dm = P.strict_dark(a[320:500, 1300:]); cnt = dm.sum(axis=1)
    rows = np.where((cnt > 12) & (cnt < 600))[0]
    nb = 320 + int(rows.max())
    dfull = P.strict_dark(a)
    rt = next(y for y in range(nb + 4, nb + 300) if dfull[y].sum() > 1500)
    m = P.ink_mask(a, 225)
    cb = int(np.where(m.sum(axis=1) > 15)[0].max())
    return W, rt - nb, H - cb

def gold_link_census(gdoc, gcode):
    cen = {}
    for i in range(gdoc.page_count):
        n = len([l for l in gdoc[i].get_links() if l.get('kind') == fitz.LINK_URI])
        if n: cen[i + 1] = n
    return cen

def photo_blob_d(gold_png, kind):
    region = {"banner": (30, 80, 620, 700), "cover": (1700, 120, 2160, 650)}[kind]
    a = P.arr(Image.open(gold_png))
    m = P.ink_mask(a[region[1]:region[3], region[0]:region[2]], 230)
    bb = P.blob_bbox(m)
    if not bb: fail(f"gold {kind} photo not measurable")
    return max(bb[2] - bb[0], bb[3] - bb[1])

def repair_flat_with_photo(src_png, gold_png, partner, dst, kind, headshot):
    a0 = P.arr(Image.open(src_png).convert('RGB'))
    r, g, b = a0[:, :, 0].astype(int), a0[:, :, 1].astype(int), a0[:, :, 2].astype(int)
    om = (np.abs(r - 158) < 25) & (np.abs(g - 134) < 25) & (np.abs(b - 64) < 28)
    disc = P.blob_bbox(om) if om.sum() > 8000 else None
    tail = partner['full'] != partner['clean']
    if not disc and not tail and not headshot:
        shutil.copy(src_png, dst); return
    work = src_png
    if disc:
        if not headshot: fail(f"{kind} shows an initials avatar but no headshot was attached")
        im = Image.open(src_png).convert('RGB')
        P.grad_fill(im, a0, disc[0] - 8, disc[1] - 8, disc[2] + 8, disc[3] + 8, gap=14, depth=18)
        work = tempfile.mktemp(suffix='.png'); im.save(work)
    if tail:
        P.repair_flat_asset(work, gold_png, partner, dst, kind, photo_action='skip')
    else:
        shutil.copy(work, dst)
    if disc:
        d_t = photo_blob_d(gold_png, kind)
        cx, cy = (disc[0] + disc[2]) // 2, (disc[1] + disc[3]) // 2
        out = Image.open(dst).convert('RGB')
        c = P.circular(headshot, d_t)
        out.paste(c, (cx - d_t // 2, cy - d_t // 2), c)
        out.save(dst)

def audit(partner, docs_in, out_dir, gold, code_pages, census, opw, opm):
    fails = []
    for doc_name, pages in code_pages.items():
        din = fitz.open(docs_in[doc_name])
        dout = fitz.open(os.path.join(out_dir, f"{partner['clean']} - {doc_name}.pdf"))
        if din.page_count != dout.page_count: fails.append(f"{doc_name}: page count changed")
        got = [i + 1 for i in range(dout.page_count) if partner['code'] in dout[i].get_text()]
        if got != pages: fails.append(f"{doc_name}: code on pages {got}, expected {pages}")
        for i in range(dout.page_count):
            t = dout[i].get_text()
            for rr in GOLD_NAME_REMNANTS:
                if rr in t: fails.append(f"{doc_name} p{i+1}: remnant '{rr}'")
            for m_ in set(re.findall(r'[A-Z]{3,}50', t)) - {partner['code']}:
                fails.append(f"{doc_name} p{i+1}: foreign code {m_}")
            links = [l for l in dout[i].get_links() if l.get('kind') == fitz.LINK_URI]
            ok = sum(1 for l in links if l['uri'] == partner['uri'] or 'mcp.prospeo' in l['uri'])
            if ok != len(links): fails.append(f"{doc_name} p{i+1}: foreign link URI")
            if len(links) != census[doc_name].get(i + 1, 0):
                fails.append(f"{doc_name} p{i+1}: {len(links)} links, expected {census[doc_name].get(i+1,0)}")
            if i + 1 not in pages and t != din[i].get_text():
                fails.append(f"{doc_name} p{i+1}: untouched page changed")
        if doc_name == 'Benchmark Report':
            w_in = set(din[0].get_text().split()); w_out = set(dout[0].get_text().split())
            miss = [w for w in w_in if w not in w_out and 'Lead' not in w and 'Test' not in w]
            if miss: fails.append(f"report p1 lost words: {miss[:4]}")
            ss = [s for b_ in dout[0].get_text('dict')['blocks'] for l in b_.get('lines', []) for s in l['spans']]
            off = [s for s in ss if 'Readers get' in s['text']]
            if not off: fails.append("report p1: offer line missing")
            else:
                ob = fitz.Rect(off[0]['bbox'])
                pix = dout[0].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(30, ob.y1, 566, ob.y1 + 14))
                arr_ = np.array(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
                rr_ = np.where(((arr_.max(axis=2)) > 45).sum(axis=1) > 900)[0]
                if len(rr_) and rr_.min() < 4: fails.append("report p1: offer touches band rule")
        din.close(); dout.close()
    fo = os.path.join(out_dir, f"{partner['clean']} - Benchmark One-Pager.png")
    w, gap, bm = gold_op_metrics(fo)
    if w != opw: fails.append(f"one-pager width {w} vs gold {opw}")
    if abs(gap - 67) > 8: fails.append(f"one-pager rule gap {gap}")
    if abs(bm - opm) > 12: fails.append(f"one-pager bottom margin {bm} vs gold {opm}")
    a = np.array(Image.open(fo).convert('RGB'))
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    if ((g > 88) & (g > r + 25) & (g > b + 12) & (r < 190))[-900:, 600:1200].sum() < 400:
        fails.append("one-pager: code ink missing in bar")
    return fails

def proof(partner, docs_in, out_dir, gold, out_jpg):
    W3 = 410
    def lab(im, t):
        im = im.resize((W3, max(1, int(im.height * W3 / im.width))))
        bar = Image.new('RGB', (W3, 18), (15, 15, 15)); ImageDraw.Draw(bar).text((4, 3), t, fill=(255, 255, 255))
        o = Image.new('RGB', (W3, im.height + 18), (255, 255, 255)); o.paste(bar, (0, 0)); o.paste(im, (0, 18)); return o
    def pc(path, pno, clip):
        d = fitz.open(path); pix = d[pno].get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=fitz.Rect(clip))
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples); d.close(); return im
    def png(path, box=None, tail=None):
        im = Image.open(path).convert('RGB')
        if tail: box = (0, im.height - tail, im.width, im.height)
        return im.crop(box)
    def fy(path, pno, needle):
        d = fitz.open(path)
        for b_ in d[pno].get_text('dict')['blocks']:
            for l in b_.get('lines', []):
                for s in l['spans']:
                    if needle in s['text']: y = s['origin'][1]; d.close(); return y
        d.close(); return 701
    def row(cells):
        h = max(c.height for c in cells)
        r_ = Image.new('RGB', (W3 * len(cells) + 8 * (len(cells) - 1), h), (255, 255, 255)); x = 0
        for c in cells: r_.paste(c, (x, 0)); x += W3 + 8
        return r_
    fA = lambda n: os.path.join(out_dir, f"{partner['clean']} - {n}")
    rows = []
    ma = fy(fA("Cold Call Openers.pdf"), 2, 'Metrics led'); mb = fy(docs_in["Cold Call Openers"], 2, 'Metrics led')
    rows.append(row([lab(pc(docs_in["Cold Call Openers"], 2, (20, mb - 15, 575, 860)), 'BEFORE cc p3 CTA'),
                     lab(pc(fA("Cold Call Openers.pdf"), 2, (20, ma - 15, 575, 860)), 'AFTER'),
                     lab(pc(gold['cc'], 2, (20, fy(gold['cc'], 2, 'Metrics led') - 15, 575, 860)), 'GOLD')]))
    ccl = fitz.open(docs_in["Cold Call Openers"]).page_count - 1
    rows.append(row([lab(pc(docs_in["Cold Call Openers"], ccl, (25, 440, 575, 860)), 'BEFORE cc closing'),
                     lab(pc(fA("Cold Call Openers.pdf"), ccl, (25, 440, 575, 860)), 'AFTER'),
                     lab(pc(gold['cc'], 12, (25, 440, 575, 860)), 'GOLD')]))
    rows.append(row([lab(pc(docs_in["Benchmark Report"], 0, (25, 90, 575, 310)), 'BEFORE report p1'),
                     lab(pc(fA("Benchmark Report.pdf"), 0, (25, 90, 575, 310)), 'AFTER'),
                     lab(pc(gold['report'], 0, (25, 90, 575, 310)), 'GOLD')]))
    ty = fy(fA("Benchmark Report.pdf"), 1, 'Try the tool')
    rows.append(row([lab(pc(docs_in["Benchmark Report"], 1, (25, ty - 75, 575, ty + 95)), 'BEFORE report p2'),
                     lab(pc(fA("Benchmark Report.pdf"), 1, (25, ty - 75, 575, ty + 95)), 'AFTER banner'),
                     lab(pc(gold['report'], 1, (25, fy(gold['report'], 1, 'Try the tool') - 75, 575,
                                                fy(gold['report'], 1, 'Try the tool') + 95)), 'GOLD')]))
    rows.append(row([lab(png(docs_in["Benchmark One-Pager"], (1100, 60, 2380, 560)), 'BEFORE one-pager header'),
                     lab(png(fA("Benchmark One-Pager.png"), (1100, 60, 2380, 560)), 'AFTER'),
                     lab(png(gold['onepager'], (1100, 60, 2380, 560)), 'GOLD')]))
    rows.append(row([lab(png(docs_in["Benchmark One-Pager"], tail=900), 'BEFORE one-pager bottom'),
                     lab(png(fA("Benchmark One-Pager.png"), tail=900), 'AFTER'),
                     lab(png(gold['onepager'], tail=900), 'GOLD')]))
    rows.append(row([lab(png(docs_in["Banner"], (40, 80, 2560, 700)), 'BEFORE banner'),
                     lab(png(fA("Banner.png"), (40, 80, 2560, 700)), 'AFTER')]))
    rows.append(row([lab(png(docs_in["Cover"], (60, 120, 2160, 900)), 'BEFORE cover'),
                     lab(png(fA("Cover.png"), (60, 120, 2160, 900)), 'AFTER')]))
    Wt = max(r_.width for r_ in rows); Ht = sum(r_.height + 8 for r_ in rows) + 30
    s = Image.new('RGB', (Wt, Ht), (255, 255, 255))
    ImageDraw.Draw(s).text((6, 6), f"{partner['clean']} ({partner['code']}) - repair proof", fill=(0, 0, 0))
    y = 26
    for r_ in rows: s.paste(r_, (0, y)); y += r_.height + 8
    s.save(out_jpg, quality=86)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--code', required=True); ap.add_argument('--audience', required=True, choices=['sales', 'agency'])
    ap.add_argument('--headshot'); ap.add_argument('--gold-zip')
    a = ap.parse_args()
    if not re.fullmatch(r'[A-Z]{3,}50', a.code): fail(f"code '{a.code}' does not match the FIRSTNAMEL50 format")
    repo = os.path.dirname(os.path.abspath(__file__))
    work = tempfile.mkdtemp()
    broken = unzip_to(a.zip, os.path.join(work, 'broken'))
    leader = unzip_to(a.gold_zip, os.path.join(work, 'leader')) if a.gold_zip else None
    full, docs = manifest(broken)
    gold = gold_paths(repo, a.audience, leader)
    guides = [g for g in docs if g in KNOWN_GUIDES]
    unknown = [g for g in docs if g not in KNOWN_GUIDES + ['Benchmark Report', 'Benchmark One-Pager', 'Banner', 'Cover']]
    if unknown: fail(f"unrecognized documents (send to Cole before shipping): {unknown}")
    pm = photo_plan(docs, a.headshot)
    partner = dict(full=full, clean=a.name, code=a.code,
                   uri=f'https://prospeo.io/pricing?coupon={a.code}',
                   photo=a.headshot, photo_mode=pm)
    gcc, grp = fitz.open(gold['cc']), fitz.open(gold['report'])
    out_dir = os.path.join(work, 'out'); os.makedirs(out_dir)
    code_pages = {}
    for g in guides:
        doc, pristine = fitz.open(docs[g]), fitz.open(docs[g])
        R.repair_guide_p3(doc, pristine, gcc, partner, f"{a.name} {g}")
        pages = [3]
        if g in CLOSING:
            R.repair_closing(doc, gcc, doc.page_count - 1, partner, f"{a.name} {g} closing")
            pages.append(doc.page_count)
        doc.save(os.path.join(out_dir, f"{a.name} - {g}.pdf"), garbage=3, deflate=True)
        doc.close(); pristine.close()
        code_pages[g] = pages
    doc, pristine = fitz.open(docs['Benchmark Report']), fitz.open(docs['Benchmark Report'])
    R.repair_report(doc, pristine, grp, partner, f"{a.name} report")
    doc.save(os.path.join(out_dir, f"{a.name} - Benchmark Report.pdf"), garbage=3, deflate=True)
    doc.close(); pristine.close()
    code_pages['Benchmark Report'] = [1, 2]
    opw, opgap, opm = gold_op_metrics(gold['onepager'])
    P.repair_onepager(docs['Benchmark One-Pager'], gold['onepager'], partner,
                      os.path.join(out_dir, f"{a.name} - Benchmark One-Pager.png"),
                      header='skip' if pm == 'keep' else 'auto', norm_gap=67, norm_margin=opm)
    for kind, nm in [('banner', 'Banner'), ('cover', 'Cover')]:
        repair_flat_with_photo(docs[nm], gold[kind], partner,
                               os.path.join(out_dir, f"{a.name} - {nm}.png"), kind, a.headshot)
    census = {}
    for g in code_pages:
        din_ = fitz.open(docs[g])
        cen = {}
        for i in range(din_.page_count):
            n = len([l for l in din_[i].get_links()
                     if l.get('kind') == fitz.LINK_URI and 'mcp.prospeo' in l.get('uri', '')])
            if n: cen[i + 1] = n
        if g == 'Benchmark Report':
            cen[1] = cen.get(1, 0) + 1; cen[2] = cen.get(2, 0) + 1
        else:
            cen[3] = cen.get(3, 0) + 1
            if g in CLOSING:
                cen[din_.page_count] = cen.get(din_.page_count, 0) + 3
        din_.close()
        census[g] = cen
    fails = audit(partner, docs, out_dir, gold, code_pages, census, opw, opm)
    pack_kind = 'Field' if a.audience == 'sales' else 'Group'
    zname = os.path.join(OUT, f"{a.name} - Prospeo {pack_kind} Pack.zip")
    with zipfile.ZipFile(zname, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)): z.write(os.path.join(out_dir, f), f)
    pj = os.path.join(OUT, f"{a.name} - Repair Proof.jpg")
    proof(partner, docs, out_dir, gold, pj)
    print(f"outputs: {zname}")
    print(f"proof:   {pj}")
    if fails:
        print("AUDIT FAIL:")
        for f in fails: print("  - " + f)
        sys.exit(1)
    print(f"AUDIT PASS: {a.name} ({a.code}), {len(guides) + 4} files repaired and verified")

if __name__ == '__main__':
    main()
