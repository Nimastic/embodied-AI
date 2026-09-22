"""Download missing reading PDFs and record verification metadata. Requires pypdf."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'papers' / 'manifest.json'


def fetch(paper):
    path = ROOT / paper['file']
    try:
        if path.exists():
            data = path.read_bytes()
        else:
            request = urllib.request.Request(paper['pdf_url'], headers={'User-Agent': 'EmbodiedAI-ReadingLibrary/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                paper['resolved_url'] = response.url
        if not data.startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        reader = PdfReader(io.BytesIO(data))
        first_page = reader.pages[0].extract_text() or ''
        normalized = ''.join(c.lower() for c in first_page if c.isalnum())
        for term in paper['verify_terms']:
            if ''.join(c.lower() for c in term if c.isalnum()) not in normalized:
                raise ValueError(f'First-page identity check failed: {term}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        paper.update(status='downloaded', pages=len(reader.pages), bytes=len(data),
                     sha256=hashlib.sha256(data).hexdigest(),
                     verified_at=datetime.now(timezone.utc).isoformat())
        paper.pop('error', None)
    except Exception as error:
        paper.update(status='unavailable', error=str(error))
    return paper


def main():
    papers = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        papers = list(pool.map(fetch, papers))
    MANIFEST.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Paper library', '', 'Named readings from the supplied conversation. Each PDF is stored once; stages link here.', '',
             'Source URLs, verification dates, byte sizes, page counts and SHA-256 hashes are in [manifest.json](manifest.json).', '',
             '| ID | Reading | Local copy | Source |', '| --- | --- | --- | --- |']
    for p in papers:
        local = f"[PDF]({p['file'].removeprefix('papers/')})" if p['status'] == 'downloaded' else 'Unavailable: see manifest'
        lines.append(f"| {p['id']} | {p['title']} ({p['year']}) | {local} | [Source]({p['source_url']}) |")
        print(p['id'], p['status'], p.get('error', f"{p.get('pages')} pages"))
    missing = [p for p in papers if p['status'] != 'downloaded']
    if missing:
        lines += ['', '## One outstanding download' if len(missing) == 1 else '## Outstanding downloads', '']
        for p in missing:
            lines.append(f"- **{p['title']}**: {p.get('access_note', p.get('error', 'Unavailable'))}")
    lines += ['', '## Version notes', '',
              '- Diffusion Policy is pinned to arXiv v4, the RSS 2023 version linked by the authors.',
              '- DreamerV3 is pinned to arXiv v1 (2023), matching the original reading recommendation.',
              '- Other unversioned arXiv URLs retrieve the version available on the download date; hashes identify the exact saved bytes.',
              '- Underactuated Robotics is course material, not a paper. Its official PDF release is dated 2023-03-24; prefer the live HTML for updated chapters and interactive examples.',
              '- DreamZero is included because it is explicitly named in the conversation. Generic WAM/VLA references do not identify additional specific papers.',
              '', 'To fetch missing files or reverify saved PDFs: `python -m pip install pypdf`, then `python scripts/download_papers.py` from the repository root.',
              'Existing PDFs are verified and retained. To deliberately update a version, change its URL and move the old file aside first.', '']
    (ROOT / 'papers' / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    if any(p['status'] != 'downloaded' for p in papers):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
