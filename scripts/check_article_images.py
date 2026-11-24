#!/usr/bin/env python3
"""Check article image files, original checksums, SVG structure and zoom wrappers."""
import argparse, hashlib, json, re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

class ArticleImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.images = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            self.images.append((a, any(t == "button" and "asb-zoom-trigger" in c.split() for t,c in self.stack)))
        if tag not in {"img","br","hr","input","meta","link","source","wbr","area","base","embed","param","track","col"}:
            self.stack.append((tag, a.get("class","")))
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

def check(root):
    public = root / "docs/public"
    findings = []
    images = {}
    pages = {}
    for page in sorted((root / "docs/posts").glob("*.md")):
        body = page.read_text(encoding="utf-8")
        parser = ArticleImages()
        parser.feed(body)
        if not parser.images:
            continue
        pages[page.stem] = []
        for a, zoom in parser.images:
            src = a.get("src","")
            if not a.get("alt","").strip(): findings.append(f"{page}: missing alt: {src}")
            if not zoom: findings.append(f"{page}: missing zoom button: {src}")
            if urlsplit(src).scheme:
                findings.append(f"{page}: remote image dependency: {src}")
                continue
            path = public / unquote(urlsplit(src).path).lstrip("/")
            if not path.is_file():
                findings.append(f"{page}: missing file: {src}")
                continue
            data = path.read_bytes()
            pages[page.stem].append(src)
            images[src] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            if path.suffix.lower() == ".svg":
                try:
                    svg = ET.fromstring(data)
                    if svg.tag != "{http://www.w3.org/2000/svg}svg": raise ValueError("not SVG")
                    if not svg.get("viewBox"): raise ValueError("missing viewBox")
                    if any(e.tag.rsplit("}",1)[-1] in {"image","script","foreignObject"} for e in svg.iter()):
                        raise ValueError("embedded raster or executable content")
                    if not svg.findtext("{http://www.w3.org/2000/svg}title"):
                        raise ValueError("missing title")
                except (ET.ParseError, ValueError) as exc:
                    findings.append(f"{path}: {exc}")
        # Markdown image references must also resolve.
        for src in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body):
            findings.append(f"{page}: image outside accessible figure wrapper: {src}")
    originals = json.loads((public/"images/figures/checksums.json").read_text())["images"]
    for im in originals:
        path = public / im["path"].lstrip("/")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != im["sha256"]:
            findings.append(f"original checksum mismatch: {im['path']}")
        if im["path"] not in images: findings.append(f"unreferenced original: {im['path']}")
    return {"ok":not findings,"pages":pages,"images":images,"originals":len(originals),"errors":findings}

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--output",type=Path)
    ap.add_argument("--live",action="store_true",help="Verify the deployed bytes of every article image")
    args=ap.parse_args()
    result=check(args.root)
    if args.live:
        import concurrent.futures, urllib.request, time
        def fetch(item):
            src, expected=item
            req=urllib.request.Request("https://quan-yi-ai.github.io"+src+"?assets="+str(time.time_ns()),
                headers={"User-Agent":"Article-Image-Check"})
            with urllib.request.urlopen(req,timeout=40) as response:
                digest=hashlib.sha256(response.read()).hexdigest()
                if digest != expected["sha256"]: raise ValueError("deployed checksum mismatch")
                return src
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(fetch,item):item[0] for item in result["images"].items()}
            result["live_images"]=[]
            for f in concurrent.futures.as_completed(futures):
                try: result["live_images"].append(f.result())
                except Exception as exc: result["errors"].append(f"{futures[f]}: {type(exc).__name__}: {exc}")
        result["ok"]=not result["errors"]
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"ok":result["ok"],"pages":len(result["pages"]),"images":len(result["images"]),
          "originals":result["originals"],"live_images":len(result.get("live_images",[])),"errors":result["errors"]},ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)
