#!/usr/bin/env python3
"""build_landing.py — assemble the floor markdown files into ONE self-contained,
offline, mobile-friendly HTML "book" you can read on a phone (or serve via nginx).

    python docs/superpowers/landing/build_landing.py            # build landing.html (EN)
    python docs/superpowers/landing/build_landing.py --ru       # build landing.ru.html (RU)

No external libraries, no CDN, no build chain — the markdown→HTML conversion happens
here in Python, so the output is a single file that opens with no network. Re-run after
editing any floor .md to regenerate. `file:line` references that are full paths become
GitHub links.

The markdown subset handled is exactly what the floor docs use: ATX headings, fenced
code blocks, pipe tables, blockquotes, ordered/unordered lists, horizontal rules,
links, inline code, bold, italic.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GH = "https://github.com/vllm-project/vllm/blob/main/"
REF_FULL = re.compile(r"^((?:vllm|tests)/[\w/().\-]+\.py):~?(\d+)$")

# (markdown file, EN title, RU title)
FLOORS = [
    ("FOUNDATIONS.md", "Floor 0 · Foundations — what an LLM is made of",
     "Этаж 0 · Основания — из чего состоит LLM"),
    ("PIPELINE-NARRATIVE.md", "Floor 1 · Why the system is shaped this way",
     "Этаж 1 · Почему система устроена именно так"),
    ("PIPELINE.md", "Floor 2 · The map (reference, file:line)",
     "Этаж 2 · Карта (справочник, file:line)"),
    ("SPEC-PP-INVARIANTS.md", "Floor 3 · Invariants & change map (spec-under-PP)",
     "Этаж 3 · Инварианты и карта изменений (spec-под-PP)"),
]

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2230;--line:#30363d;--fg:#e6edf3;
--muted:#9aa4b2;--accent:#58a6ff;--accent2:#3fb950;--code:#79c0ff;--spec:#bc8cff;}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
overflow-wrap:break-word;word-wrap:break-word}
.wrap{max-width:820px;margin:0 auto;padding:0 16px 120px}
header.top{padding:22px 16px 14px;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,#11161f,#0d1117)}
header.top h1{margin:0 0 4px;font-size:21px}
header.top p{margin:0;color:var(--muted);font-size:13.5px}
nav.toc{position:sticky;top:0;z-index:30;display:flex;gap:6px;overflow-x:auto;
padding:9px 12px;background:#0d1117f2;backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);-webkit-overflow-scrolling:touch}
nav.toc a{flex:0 0 auto;color:var(--muted);text-decoration:none;font-size:13px;
padding:6px 11px;border:1px solid var(--line);border-radius:999px;white-space:nowrap}
nav.toc a:active,nav.toc a.on{color:var(--fg);background:var(--panel2);border-color:var(--accent)}
section.floor{padding-top:26px;border-top:1px dashed var(--line);margin-top:26px}
section.floor:first-of-type{border-top:0}
.floortag{display:inline-block;font-size:12px;color:var(--spec);border:1px solid #3a2d55;
background:#1a1430;border-radius:999px;padding:2px 10px;margin-bottom:8px}
h1,h2,h3,h4{line-height:1.3;margin:1.4em 0 .5em}
h1{font-size:24px} h2{font-size:20px;border-left:3px solid var(--accent);padding-left:10px}
h3{font-size:17px;color:#cdd6e0} h4{font-size:15px;color:var(--muted)}
a{color:var(--accent)}
p{margin:.7em 0}
ul,ol{padding-left:1.4em;margin:.6em 0}
li{margin:.3em 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.86em;
background:#11202f;border:1px solid #1d3650;border-radius:4px;padding:.5px 5px;color:var(--code)}
a.ref{white-space:nowrap}
pre{background:#0a0e14;border:1px solid var(--line);border-radius:10px;padding:13px;
overflow-x:auto;-webkit-overflow-scrolling:touch;font-size:12.5px;line-height:1.45}
pre code{background:none;border:0;padding:0;color:#b9c4d0;font-size:inherit}
blockquote{margin:.9em 0;padding:.4em 14px;border-left:3px solid #2d4d6e;background:#0c1622;
border-radius:0 8px 8px 0;color:#c6d3e2}
hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.8em 0;
border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);border-right:1px solid var(--line);
padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--panel2);color:#cdd6e0;font-weight:600}
tr:last-child td{border-bottom:0} th:last-child,td:last-child{border-right:0}
strong{color:#fff} em{color:#d7c8ef}
footer{color:var(--muted);font-size:12px;text-align:center;padding:30px 16px}
@media(max-width:520px){body{font-size:15.5px}header.top h1{font-size:18px}}
"""

JS = """
const links=[...document.querySelectorAll('nav.toc a')];
const secs=links.map(a=>document.querySelector(a.getAttribute('href')));
function onScroll(){let i=secs.length-1;for(let k=0;k<secs.length;k++){
 if(secs[k].getBoundingClientRect().top<120)i=k;}
 links.forEach((a,k)=>a.classList.toggle('on',k===i));}
document.addEventListener('scroll',onScroll,{passive:true});onScroll();
"""


def slug(s: str) -> str:
    return "f" + re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fmt_inline(text: str) -> str:
    """Inline markdown → HTML. Pull code spans out first so their content is verbatim."""
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00C{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    def unstash(m: re.Match) -> str:
        raw = codes[int(m.group(1))]
        esc = html.escape(raw, quote=True)
        ref = REF_FULL.match(raw)
        if ref:
            url = GH + ref.group(1) + "#L" + ref.group(2)
            return f'<a class="ref" href="{url}"><code>{esc}</code></a>'
        return f"<code>{esc}</code>"

    return re.sub(r"\x00C(\d+)\x00", unstash, text)


def parse_blocks(lines: list[str]) -> str:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        # fenced code
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        # blank
        if not line.strip():
            i += 1
            continue
        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{fmt_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # horizontal rule
        if re.match(r"^---+\s*$", line):
            out.append("<hr>")
            i += 1
            continue
        # table (header row then |---| separator)
        if line.lstrip().startswith("|") and i + 1 < n and re.match(
                r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{fmt_inline(c)}</th>" for c in header)
            body = ""
            for r in rows:
                tds = "".join(f"<td>{fmt_inline(c)}</td>" for c in r)
                body += f"<tr>{tds}</tr>"
            out.append(f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
                       f"<tbody>{body}</tbody></table></div>")
            continue
        # blockquote (collect consecutive > lines, recurse)
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>" + parse_blocks(buf) + "</blockquote>")
            continue
        # unordered / ordered list (markers at column 0; indented lines = continuation)
        list_marker = None
        if re.match(r"^[-*]\s+", line):
            list_marker, wrap = r"^[-*]\s+", "ul"
        elif re.match(r"^\d+\.\s+", line):
            list_marker, wrap = r"^\d+\.\s+", "ol"
        if list_marker:
            items = []
            while i < n and re.match(list_marker, lines[i]):
                text = re.sub(list_marker, "", lines[i])
                i += 1
                while i < n and lines[i].strip() and lines[i][0] in " \t":
                    text += " " + lines[i].strip()
                    i += 1
                items.append(text)
            out.append(f"<{wrap}>"
                       + "".join(f"<li>{fmt_inline(t)}</li>" for t in items)
                       + f"</{wrap}>")
            continue
        # paragraph (gather until blank/structural)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|```|>|\s*[-*]\s|\s*\d+\.\s|---+\s*$)", lines[i]) \
                and not lines[i].lstrip().startswith("|"):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + fmt_inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


def md_to_html(md: str) -> str:
    return parse_blocks(md.split("\n"))


def build(ru: bool) -> Path:
    lang = "ru" if ru else "en"
    title = ("Пайплайн исполнения vLLM V1 — книга" if ru
             else "vLLM V1 execution pipeline — the book")
    intro = ("Четыре «этажа» в одном файле для офлайн-чтения: основания → почему так "
             "устроено → карта → инварианты. Ссылки file:line ведут на GitHub."
             if ru else
             "Four floors in one offline file: foundations → why it's shaped this way → "
             "the map → invariants. file:line links go to GitHub.")
    toc, body = [], []
    for fname, en_t, ru_t in FLOORS:
        src_name = fname.replace(".md", ".ru.md") if ru else fname
        src = HERE / src_name
        if not src.exists():
            if ru and (HERE / fname).exists():
                print(f"  warn: {src_name} missing — falling back to EN {fname}")
                src, src_name = HERE / fname, fname
            else:
                print(f"  skip (missing): {src_name}")
                continue
        chap = ru_t if ru else en_t
        sid = slug(fname)
        toc.append(f'<a href="#{sid}">{html.escape(chap)}</a>')
        inner = md_to_html(src.read_text(encoding="utf-8"))
        tag = ("источник" if ru else "source")
        body.append(
            f'<section class="floor" id="{sid}">'
            f'<div class="floortag">{html.escape(chap)} · {tag}: {src_name}</div>'
            f"{inner}</section>")
    page = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p></header>
<nav class="toc">{''.join(toc)}</nav>
<main class="wrap">
{''.join(body)}
<footer>Generated by build_landing.py from the floor .md files · read-only research · not committed</footer>
</main>
<script>{JS}</script>
</body>
</html>
"""
    out = HERE / (f"landing.ru.html" if ru else "landing.html")
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    ru = "--ru" in sys.argv[1:]
    path = build(ru)
    size = path.stat().st_size
    print(f"wrote {path}  ({size // 1024} KB)")
