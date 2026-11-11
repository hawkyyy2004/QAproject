"""Corpus-only reading products. Never rewrite original QA evidence."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

REVISION = 'corpus-20261111'


class Article(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[]; self.blocks=[]; self.images=[]; self.buffer=[]; self.scope=0
        self.caption=''; self.figure_images=[]; self.headings=[]; self.level=0

    def flush(self):
        text=' '.join(''.join(self.buffer).split()); self.buffer=[]
        if text:
            self.blocks.append(dict(text=text, level=self.level, scope=self.scope))

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        parent=self.stack[-1] if self.stack else ('',False,0)
        excluded=parent[1] or tag in {'script','style','nav','footer','noscript','template','svg','head'}
        excluded=excluded or attrs.get('role') in {'navigation','banner','contentinfo'}
        excluded=excluded or bool(re.search(r'(?:^|[ _-])(advertisement|cookie-banner|social-share)(?:$|[ _-])',attrs.get('class',''),re.I))
        scope=parent[2]+int(tag in {'article','main'} or attrs.get('role')=='main')
        if tag in {'p','div','section','article','main','li','tr','figure','figcaption','h1','h2','h3','h4','h5','h6'}:
            self.flush()
        self.scope=scope
        if re.fullmatch('h[1-6]',tag):self.level=int(tag[1])
        if tag=='figure':self.figure_images=[]
        if tag=='img' and not excluded:
            src=attrs.get('src') or attrs.get('data-src') or ''
            if src and not re.search(r'logo|icon|avatar|tracking|pixel',src+' '+attrs.get('class',''),re.I):
                self.images.append(dict(src=src,alt=attrs.get('alt',''),caption='',scope=scope))
                self.figure_images.append(len(self.images)-1)
        if tag in {'td','th'} and not excluded:self.buffer.append(' | ')
        if tag=='br' and not excluded:self.buffer.append('\n')
        if tag not in {'img','br','hr','meta','link','input','source','wbr','area','base','embed','param','track','col'}:
            self.stack.append((tag,excluded,scope))

    def handle_endtag(self,tag):
        if tag=='figcaption':
            caption=' '.join(''.join(self.buffer).split())
            for i in self.figure_images:self.images[i]['caption']=caption
        if tag in {'p','div','section','article','main','li','tr','figure','figcaption','h1','h2','h3','h4','h5','h6'}:self.flush()
        if re.fullmatch('h[1-6]',tag):self.level=0
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:
                del self.stack[i:];break
        self.scope=self.stack[-1][2] if self.stack else 0

    def handle_data(self,data):
        if not self.stack or not self.stack[-1][1]:self.buffer.append(data)


def article(raw, url='', charset=None):
    from .web_sources import decode_html
    parser=Article();parser.feed(decode_html(raw,charset));parser.close();parser.flush()
    scoped=any(b['scope'] for b in parser.blocks)
    blocks=[];removed=[];seen={}
    for index,b in enumerate(parser.blocks):
        text=b['text'];reason=''
        if scoped and not b['scope']:reason='outside-main-content'
        elif re.fullmatch(r'loading[.…\s]*|加载中[。.…\s]*|click here|点击这里|advertisement|广告',text,re.I):reason='placeholder'
        elif text in seen and len(text)>=40:reason='exact-repeated-paragraph'
        if reason:removed.append(dict(index=index,text=text,reason=reason));continue
        seen[text]=index;blocks.append(b)
    images=[];seen_urls=set()
    for im in parser.images:
        target=urljoin(url,im['src'])
        if scoped and not im['scope']:continue
        if urlsplit(target).scheme not in {'http','https'} or target in seen_urls:continue
        seen_urls.add(target);images.append(dict(url=target,alt=im['alt'],caption=im['caption']))
    return dict(blocks=blocks,images=images,removed=removed,
                text='\n\n'.join(b['text'] for b in blocks))


def page_structure(page,number,textpage=None):
    import pymupdf
    blocks=[]
    data=page.get_text('dict',flags=pymupdf.TEXTFLAGS_TEXT,textpage=textpage)
    for block in data['blocks']:
        for line in block.get('lines',[]):
            spans=line['spans'];text=''.join(span['text'] for span in spans).strip()
            if text:blocks.append(dict(text=text,bbox=list(line['bbox']),size=max(span['size'] for span in spans),bold=any(span['flags'] & 16 for span in spans)))
    tables=[]
    try:
        for table in page.find_tables().tables:
            rows=table.extract()
            if len(rows)>1 and len(rows[0])>1:tables.append(dict(bbox=list(table.bbox),rows=rows))
    except (ValueError,RuntimeError):pass
    return dict(page=number,height=page.rect.height,lines=blocks,tables=tables)


def pdf_structure(path, heartbeat=lambda:None):
    import pymupdf
    pages=[]
    with pymupdf.open(path) as pdf:
        for number,page in enumerate(pdf,1):
            heartbeat();pages.append(page_structure(page,number))
    return finish_structure(pages)


def finish_structure(pages):
    sizes=Counter()
    for page in pages:
        for line in page['lines']:sizes[round(line['size'])]+=len(line['text'])
    body=sizes.most_common(1)[0][0] if sizes else 0
    margins=Counter()
    for page in pages:
        margins.update(set(line['text'] for line in page['lines'] if line['bbox'][3]<40 or line['bbox'][1]>page['height']-35))
    for page in pages:
        for line in page['lines']:
            line['level']=(1 if line['size']>=body*1.45 else 2) if body and line['size']>=body*1.18 and len(line['text'])<180 else 0
            line['margin']=(line['bbox'][3]<40 or line['bbox'][1]>page['height']-35) and (margins[line['text']]>=3 or bool(re.fullmatch(r'(?:Page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?',line['text'],re.I)))
    return dict(engine='PyMuPDF structured text/table extraction',pages=pages,
                limitations=['扫描页保留原图，未冒充OCR正文','字体推断标题需核对；公式保留可提取字符，不宣称完整LaTeX恢复'])


def markdown(title, units, structure=None):
    lines=['# '+title.replace('\n',' '),'']
    if structure:
        for page in structure['pages']:
            lines.extend([f'<!-- page-{page["page"]:02d} -->',''])
            for line in page['lines']:
                if line.get('margin'):continue
                box=line['bbox']
                if any(box[0]>=t['bbox'][0]-1 and box[1]>=t['bbox'][1]-1 and box[2]<=t['bbox'][2]+1 and box[3]<=t['bbox'][3]+1 for t in page['tables']):continue
                lines.extend([('#'*min(6,line['level']+1)+' ' if line['level'] else '')+line['text'],''])
            for table in page['tables']:
                rows=[[str(c or '').replace('|','\\|').replace('\n','<br>') for c in row] for row in table['rows']]
                lines.extend(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |'])
                lines.extend('| '+' | '.join(row)+' |' for row in rows[1:]);lines.append('')
            for u in units:
                if u['page']==page['page'] and u['image']:
                    lines.extend([f'![第{u["page"]}页完整原图](images/{Path(u["image"]).name})',''])
    else:
        for u in units:
            if not u['image']:lines.extend([('#'*min(6,u.get('heading_level',0)+1)+' ' if u.get('heading_level') else '')+u['text'],''])
            else:lines.extend([f'![原图](images/{Path(u["image"]).name})',u.get('caption',''),''])
    return '\n'.join(lines)


def source_row(facts, units, url='', title=''):
    """One canonical seven-column export for both entry and Corpus downloads."""
    def unique(values):return list(dict.fromkeys(v for v in values if v))
    row=dict(facts[0]);text='\n\n'.join(unique(u['text'] for u in units if not u.get('image') and u.get('text','').strip()))
    row.update(entity_description='\n\n'.join(unique(f['entity_description'] for f in facts)),
               full_text=text,full_text_length=len(text))
    for key,value in [('page_urls',url),('page_titles',title)]:
        row[key]=value or '|||'.join(unique(v for f in facts for v in f[key].split('|||')))
    return row


def mineru_structure(store,doc,heartbeat):
    """Optional local MinerU JSON, without invoking the legacy crop pipeline."""
    import json
    import subprocess
    from . import layout
    from .settings import settings,ROOT
    from .console import child_env
    config=settings();block=config.get('mineru') or {}
    folder=store.root/'corpus/mineru'/doc['content_hash'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/'result.json'
    historical=store.root/'layout'/doc['content_hash']/'mineru.json'
    if not target.is_file() and historical.is_file():target=historical
    if not target.is_file():
        python=layout.runner(config)
        if not python:raise ValueError('未找到可用MinerU；请检查原有解析环境或配置，不会把本次请求冒充为MinerU成功。')
        heartbeat()
        try:
            run= subprocess.run([python,str(ROOT/'scripts/mineru_parse_worker.py'),doc['path'],str(target),str(block.get('tier') or 'basic')],
                capture_output=True,env=child_env(),timeout=int(block.get('timeout_seconds') or 900))
        except subprocess.TimeoutExpired:
            target.unlink(missing_ok=True);raise ValueError('MinerU解析超时；任务可重试，未将不完整结果当作成功。') from None
        heartbeat()
        if run.returncode or not target.is_file():
            target.unlink(missing_ok=True);raise ValueError('MinerU解析失败；保留原件，未静默改用其他解析器。')
    raw=json.loads(target.read_text(encoding='utf-8'))
    def flatten(value):
        if isinstance(value,str):return value
        if isinstance(value,list):return ''.join(flatten(i.get('content','')) for i in value if isinstance(i,dict))
        return ''
    pages=[]
    for page in raw.get('pages',[]):
        lines=[]
        for block in page.get('blocks',[]):
            text=flatten(block.get('content')).strip()
            if not text:continue
            kind=block.get('type','text')
            if kind in {'equation','interline_equation'}:text='$$\n'+text+'\n$$'
            lines.append(dict(text=text,bbox=block.get('bbox',[0,0,0,0]),
                              level=1 if kind in {'title','heading'} else 0,kind=kind,margin=False))
        pages.append(dict(page=int(page.get('page_idx',0))+1,lines=lines,tables=[]))
    if not pages:raise ValueError('MinerU没有返回可用页面。')
    return dict(engine='MinerU (local)',pages=pages,
                limitations=['保留完整页图，不裁剪','公式/表格/标题保留MinerU原输出，仍须核对识别质量'])


def configured_engine(store,doc,config,requested=None):
    """Respect the existing auto/manifest/off policy instead of overriding it."""
    from . import layout
    if requested in {'native','mineru'}:return requested,''
    mode=str(config.get('pdf_layout','auto')).lower()
    if mode=='off':return 'native','沿用原有pdf_layout=off配置。'
    cached=(store.root/'layout'/doc['content_hash']/'mineru.json').is_file() or (store.root/'corpus/mineru'/doc['content_hash']/'result.json').is_file()
    if cached or (mode=='auto' and layout.runner(config)):return 'mineru','沿用原有MinerU配置或已保存的MinerU结果。'
    return 'native','沿用原有自动/清单模式的原生兜底：本次未找到可用MinerU或匹配清单；不是MinerU解析成功。'


def structured_units(doc,structure,images):
    from .parsers import text_spans
    from .store import digest
    result=[dict(u) for u in images if u.get('image')]
    for page in structure['pages']:
        n=0
        for line in page['lines']:
            if line.get('margin'):continue
            for start,end in text_spans(line['text']):
                n+=1;text=line['text'][start:end]
                u=dict(id=f'{doc["id"]}:mineru:p{page["page"]}:s{n}',text=text,page=page['page'],seg=n,
                       locator=f'page-{page["page"]:02d}-seg-{n:02d}',image='',asset_id='',
                       language='zh' if re.search('[\u4e00-\u9fff]',text) else 'en',heading_level=line['level'])
                u['hash']=digest(u);result.append(u)
    return result
