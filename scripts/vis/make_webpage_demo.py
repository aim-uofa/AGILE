"""Write an offline English reference/overlay/render contact sheet."""
from html import escape
from pathlib import Path


def write_page(reports, output):
    sections=[]
    for report in reports:
        sections.append('<h1>'+escape(str(report.get('label','AGILE reconstruction')))+' — reconstruction</h1>')
        sections.append('<p>Saved final geometry. Reference, overlay and rendered mesh share the same camera.</p>')
        for notice in report.get('notices',[]):
            sections.append('<p>'+escape(str(notice))+'</p>')
        if report.get('video'):
            sections.append('<video controls preload="metadata" src="'+escape(report['video'],quote=True)+'"></video>')
        for frame in report['frames']:
            sections.append('<h2>Frame '+escape(str(frame['frame_id']))+'</h2><div class="row">')
            for key in ('reference','overlay','render'):
                sections.append('<figure><figcaption>'+key.title()+'</figcaption><img loading="lazy" src="'+escape(frame[key],quote=True)+'" alt="'+key+'"></figure>')
            sections.append('</div>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AGILE reconstruction</title>'
    page+='<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f5f6f7;color:#15202c}.row{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img,video{width:100%;height:auto}figcaption{padding:8px 0}h2{margin-top:36px}@media(max-width:640px){.row{display:block}}</style><body>'
    Path(output).write_text(page+'\n'.join(sections)+'</body></html>')
