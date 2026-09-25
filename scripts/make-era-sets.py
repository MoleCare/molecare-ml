"""Split the ISIC evaluation set into oldest and newest thirds by isic_id.

ISIC ids are assigned in accession order, so the low ids are the 2016-era
challenge images that any training pull would almost certainly have swept up,
and the high ids are recent additions. If the model scores much better on the
old third than the new third, that difference is evidence it saw some of them
in training rather than evidence it generalises.

Both thirds keep their own class balance, so sensitivity and specificity stay
comparable between them.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
SRC = {"Melanoma": ROOT / "isic-raw/mel-all", "NotMelanoma": ROOT / "isic-raw/ben-all"}
ID = re.compile(r"ISIC_(\d+)")


def isic_number(path: pathlib.Path) -> int:
    m = ID.search(path.name)
    return int(m.group(1)) if m else -1


for era in ("oldest", "newest"):
    for cls in SRC:
        (ROOT / f"era-{era}" / cls).mkdir(parents=True, exist_ok=True)

counts = {}
for cls, src in SRC.items():
    images = sorted((p for p in src.glob("*.jpg")), key=isic_number)
    third = len(images) // 3
    for era, chunk in (("oldest", images[:third]), ("newest", images[-third:])):
        dest = ROOT / f"era-{era}" / cls
        for p in chunk:
            link = dest / p.name
            if not link.exists():
                link.symlink_to(p)
        counts[f"{era}/{cls}"] = len(chunk)
        lo, hi = isic_number(chunk[0]), isic_number(chunk[-1])
        print(f"  {era:7} {cls:12} {len(chunk):6} images  ISIC_{lo:07d}..ISIC_{hi:07d}")
