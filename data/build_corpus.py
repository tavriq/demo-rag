"""Build data/corpus.jsonl from the Labour Code of the Russian Federation.

Source: per-article pages of consultant.ru (public, allowed by robots.txt).
Only the standard library is used. Requests are sequential with a 1-2 s pause.

Usage:
    python3 data/build_corpus.py --cache /tmp/tk-cache --out data/corpus.jsonl
    python3 data/build_corpus.py --cache /tmp/tk-cache --offline   # re-parse only

Raw HTML is cached, so a re-run only parses. The script does not bypass any
captcha or anti-bot protection: if a page has no document body, it is reported
and skipped.
"""
import argparse
import html
import json
import os
import random
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

BASE = "https://www.consultant.ru"
TOC_URL = BASE + "/document/cons_doc_LAW_34683/"
UA = "Mozilla/5.0 (compatible; demo-rag corpus builder)"

# Paragraph that continues the previous part: "1)", "1.1)", "а)", "- ..."
POINT = re.compile(r"^(\d+(\.\d+)*\)|[а-я]\)|[а-я]\.\d*\)|-\s)")


def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_toc(page):
    """Return live articles in code order with their chapter and section."""
    links = re.findall(r'href="(/document/cons_doc_LAW_34683/[0-9a-f]+/)"[^>]*>([^<]{0,400})', page)
    toc, seen, chapter, section = [], set(), None, None
    for href, text in links:
        text = html.unescape(text).strip()
        if href in seen:
            continue
        seen.add(href)
        if text.startswith("Глава"):
            chapter = text
            continue
        if text.startswith("Раздел"):
            section = text
            continue
        m = re.match(r"Статья (\d+(?:\.\d+)?(?:-\d+)?)\.\s*(.*)", text)
        if not m or m.group(2).lower().startswith("утратил"):
            continue
        toc.append({"url": BASE + href, "article": m.group(1), "chapter": chapter, "section": section})
    return toc


def edition_date(page):
    m = re.search(r"\(ред\. от (\d\d)\.(\d\d)\.(\d{4})\)", page)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


class ArticleParser(HTMLParser):
    """Collect normative paragraphs of the article body.

    Skipped: editorial notes ("в ред. Федерального закона ..."), links to
    previous editions, publisher inserts and comment blocks.
    """

    SKIP = ("document__insert", "document__edit", "doc-roll")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_content = False
        self.depth = 0
        self.skip_depth = None
        self.cur = None
        self.paras = []
        self.h1 = None
        self.in_h1 = False
        self.h1buf = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if not self.in_content:
            if tag == "div" and "document-page__content" in cls:
                self.in_content, self.depth = True, 1
            return
        if tag == "div":
            self.depth += 1
            if self.skip_depth is None and any(s in cls for s in self.SKIP):
                self.skip_depth = self.depth
        if tag == "h1":
            self.in_h1, self.h1buf = True, []
        if tag == "p" and self.skip_depth is None and not self.in_h1:
            self.cur = []

    def handle_endtag(self, tag):
        if not self.in_content:
            return
        if tag == "h1":
            self.in_h1, self.h1 = False, "".join(self.h1buf)
        if tag == "p" and self.cur is not None:
            self.paras.append("".join(self.cur))
            self.cur = None
        if tag == "div":
            if self.skip_depth == self.depth:
                self.skip_depth = None
            self.depth -= 1
            if self.depth == 0:
                self.in_content = False

    def handle_data(self, data):
        if not self.in_content:
            return
        if self.in_h1:
            self.h1buf.append(data)
        elif self.cur is not None and self.skip_depth is None:
            self.cur.append(data)


def clean(s):
    return re.sub(r"[ \t]+", " ", s.replace("\xa0", " ")).strip()


def parse_article(page):
    """Return (number, title, text). Parts are numbered "1. ", "2. ", ...

    A paragraph starting with an uppercase letter opens a new part; points
    ("1)", "а)"), dashes, lowercase paragraphs and items after a line ending
    with ":" or ";" (e.g. the list of holidays in art. 112) belong to the
    current part.
    Parts that lost force stay in place ("N. Часть ... утратила силу"), so the
    numbering matches the official one; a joint placeholder for several parts
    gets a range ("3-4. Части третья - четвертая утратили силу").
    A trailing "Примечание" is kept as is.
    """
    p = ArticleParser()
    p.feed(page)
    head = re.sub(r"^ТК РФ\s+", "", clean(p.h1 or ""))
    m = re.match(r"Статья\s+(\d+(?:\.\d+)?(?:-\d+)?)\.\s*(.*)", head)
    if not m:
        raise ValueError("no article heading")
    parts, notes = [], []
    for para in filter(None, (clean(x) for x in p.paras)):
        if para.startswith("Примечание") or notes:
            notes.append(para)
        elif parts and (POINT.match(para) or para[0].islower() or para[0] in "-–—"
                        or parts[-1].rstrip().endswith((":", ";"))):
            parts[-1] += "\n" + para
        else:
            parts.append(para)
    lines, n = [], 0
    for t in parts:
        span = lost_parts_span(t)
        if span and span[0] == n + 1 and span[1] > span[0]:
            lines.append(f"{span[0]}-{span[1]}. {t}")  # "Части третья - четвертая утратили силу"
            n = span[1]
        else:
            n += 1
            lines.append(f"{n}. {t}")
    text = "\n".join(lines)
    if notes:
        text += "\n" + "\n".join(notes)
    return m.group(1), m.group(2).strip(), text


ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7,
    "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12,
    "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16,
    "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20,
}


def ordinal(word):
    """"третья" -> 3; longest stem wins ("двенадцат" before "двадцат")."""
    for stem in sorted(ORDINALS, key=len, reverse=True):
        if word.startswith(stem):
            return ORDINALS[stem]
    return None


def lost_parts_span(part):
    """(3, 4) for "Части третья - четвертая утратили силу ...", else None."""
    m = re.match(r"Части ([а-я]+) [-–и] ([а-я]+) утратили силу", part)
    if not m:
        return None
    a, b = ordinal(m.group(1)), ordinal(m.group(2))
    return (a, b) if a and b else None


def cached(url, path, pause, offline=False):
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return open(path, encoding="utf-8").read()
    if offline:
        raise FileNotFoundError("not in cache (offline)")
    page = fetch(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    time.sleep(pause + random.random())
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="tk-cache")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "corpus.jsonl"))
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--offline", action="store_true", help="parse cached pages only")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    toc_page = cached(TOC_URL, os.path.join(args.cache, "toc.html"), args.pause, args.offline)
    ed = edition_date(toc_page)
    toc = parse_toc(toc_page)
    print(f"articles in TOC: {len(toc)}, edition: {ed}", file=sys.stderr)

    rows, skipped = [], []
    for a in toc:
        try:
            page = cached(a["url"], os.path.join(args.cache, f"{a['article']}.html"), args.pause, args.offline)
            if "document-page__content" not in page:
                raise ValueError("no document body")
            num, title, text = parse_article(page)
            if num != a["article"]:
                raise ValueError(f"heading mismatch {num}")
        except Exception as e:  # report and go on, never retry aggressively
            skipped.append((a["article"], str(e)[:80]))
            continue
        rows.append({
            "id": f"TK-{num}", "article": num, "title": title,
            "chapter": a["chapter"], "section": a["section"], "text": text,
            "source_url": a["url"], "edition_date": ed,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {len(rows)} articles to {args.out}; skipped {len(skipped)}: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
