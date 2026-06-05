# Read it on your phone — deploy options

The reading surface is **`landing.html`** (the offline "book": all four floors in one
self-contained file) and **`landing.ru.html`** (Russian). They need **no network** at
runtime — every dependency is inlined. Pick whichever delivery fits.

## Option A — just the file (simplest, no server)

`landing.html` is fully self-contained. You can:
- **AirDrop / Telegram "Saved Messages" / email it to yourself** and open the file
  directly in the phone browser. Works offline, on a plane, anywhere.
- The `file:line` chips are links to GitHub (`vllm-project/vllm`), so those need net;
  everything else reads offline.

```bash
# regenerate the books from the floor .md sources first (do this after any edit):
python3 docs/superpowers/landing/build_landing.py        # landing.html  (EN)
python3 docs/superpowers/landing/build_landing.py --ru   # landing.ru.html (RU)
```

## Option B — Docker + nginx (read over headscale, e.g. on gpu-wb)

```bash
# from the repo root:
docker build -t vllm-pipeline-book docs/superpowers/landing
docker run -d --name vllm-book -p 8080:80 vllm-pipeline-book
# → http://<host>:8080/         (the book, landing.html)
# → http://<host>:8080/landing.ru.html
# → http://<host>:8080/index.html   (the interactive landing)
# → http://<host>:8080/             autoindex lists every file incl. the .md sources
```

On **gpu-wb** (you reach it over headscale): rsync this folder up, build, run, then open
`http://gpu-wb:8080/` from your phone on the tailnet. Don't disturb the prod GPUs — this
is a static nginx container, it uses no GPU.

```bash
rsync -az docs/superpowers/landing/ gpu-wb:/root/pipeline-book/
ssh gpu-wb 'cd /root/pipeline-book && docker build -t vllm-pipeline-book . && \
  docker rm -f vllm-book 2>/dev/null; docker run -d --name vllm-book -p 8080:80 vllm-pipeline-book'
# phone (on headscale): http://gpu-wb:8080/
```

Stop / clean up:
```bash
docker rm -f vllm-book
```

## Option C — no Docker, just Python (quick local/LAN serve)

```bash
python3 -m http.server -d docs/superpowers/landing 8080
# → http://<this-machine>:8080/landing.html
```

## What's in the folder

| File | What |
|---|---|
| `landing.html` / `landing.ru.html` | the **book** — all 4 floors, offline, mobile-first (deploy this) |
| `index.html` / `index.ru.html` | the **interactive** landing (clickable diagram, stepper, glossary) |
| `FOUNDATIONS.md` | Floor 0 — what an LLM is made of |
| `PIPELINE-NARRATIVE.md` | Floor 1 — why the system is shaped this way |
| `PIPELINE.md` | Floor 2 — the reference map (`file:line`) |
| `SPEC-PP-INVARIANTS.md` | Floor 3 — invariants / change map for spec-under-PP |
| `*.ru.md` | Russian versions of the floors |
| `build_landing.py` | regenerates `landing.html` from the floor `.md` files |
| `check_refs.py` | validates every `file:line` reference resolves (anti-rot) |
