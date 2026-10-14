"""Synthetic, locally constructed evidence. Never inserted into the live workspace."""
from pathlib import Path
import time

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from .parsers import extract
from .settings import source_ref, settings
from .store import uid, encode, digest, normalized

SAMPLES = [
    {'name': '近岸叶绿素的季节观测.pdf', 'topic': '海洋生态', 'keywords': ['叶绿素', '浮游植物', '季节变化'],
     'summary': '用合成月度记录展示近岸叶绿素变化，比较四个观测期的浓度及其变化范围。',
     'evidence': ['演示海域 A 在 2025 年 1 至 4 月的叶绿素浓度为 2、4、7、5 mg/m3。',
                  '这些数字为合成演示记录，不是实际海域观测数据。',
                  '仅比较上述四期，不据此推断其他时期的生态变化或因果关系。'],
     'values': [2,4,7,5], 'labels': ['Jan','Feb','Mar','Apr'], 'unit': 'mg/m3',
     'qas': [('在演示海域 A 的 2025 年 1 至 4 月合成记录中，叶绿素浓度最高的是哪个月份？','3 月最高，为 7 mg/m3。'),
             ('演示海域 A 的四期合成叶绿素浓度呈现怎样的变化？','1 至 3 月由 2 升至 7 mg/m3，4 月降至 5 mg/m3。'),
             ('演示海域 A 的四期叶绿素记录可以支持对其他海域的因果判断吗？','不能。资料明确限定为合成记录，仅比较四期变化，不推断其他海域或因果关系。')]},
    {'name': '海上风电的成本情景比较.pdf', 'topic': '海洋能源', 'keywords': ['海上风电', '成本评价', '情景比较'],
     'summary': '对四个合成风电方案的单位成本进行比较，保留同一口径和情景限制。',
     'evidence': ['四个演示方案 A、B、C、D 的单位供电成本分别为 0.8、0.6、0.7、0.9 元/kWh。',
                  '数字为合成情景结果，各方案采用同一成本口径，仅用于展示比较。',
                  '这些结果不是实际项目报价，也不能直接推广到其他风电项目。'],
     'values': [.8,.6,.7,.9], 'labels': ['A','B','C','D'], 'unit': 'yuan/kWh',
     'qas': [('四个同口径合成风电方案中，哪个方案的单位供电成本最低？','方案 B 最低，为 0.6 元/kWh。'),
             ('在演示风电成本情景中，方案 A 和方案 C 的单位供电成本分别是多少？','A 为 0.8 元/kWh，C 为 0.7 元/kWh，采用相同成本口径。'),
             ('合成风电方案的成本比较结果适用于哪些范围？','只适用于给出的合成情景和统一成本口径，不是实际项目报价，也不能直接推广到其他项目。')]},
    {'name': '正弦函数的采样示例.pdf', 'topic': '数学', 'keywords': ['三角函数', '正弦函数', '周期'],
     'summary': '展示单位圆中正弦函数的定义及四个角度的函数值，作为细分检索方向归入数学。',
     'evidence': ['单位圆上角度对应点的纵坐标定义为该角的正弦值。',
                  '本示例给出的角度为 0、90、180、270 度，正弦值分别为 0、1、0、-1。',
                  '正弦函数以 360 度为一个周期，示例用于教学演示。'],
     'values': [0,1,0,-1], 'labels': ['0','90','180','270'], 'unit': 'sin(angle)',
     'qas': [('单位圆中，正弦值对应角度所确定点的哪个坐标？','对应该点的纵坐标。'),
             ('在给出的正弦采样示例中，90 度和 270 度的正弦值分别是多少？','分别为 1 和 -1。'),
             ('正弦函数以角度表示的周期是多少？','360 度。')]},
    {'name': '椭圆的几何参数示例.pdf', 'topic': '数学', 'keywords': ['圆锥曲线', '椭圆', '几何参数'],
     'summary': '给出中心在原点、横向半轴为 3、纵向半轴为 2 的椭圆示例，检索方向与三角函数分开。',
     'evidence': ['本教学示例椭圆的方程为 x^2/9 + y^2/4 = 1，中心位于原点。',
                  '横向半轴长度为 3，纵向半轴长度为 2，长轴沿 x 轴方向。',
                  '图中柱形仅比较两个半轴及其全轴长度，不是椭圆曲线本身。'],
     'values': [3,2,6,4], 'labels': ['a','b','2a','2b'], 'unit': 'length',
     'qas': [('对于 x^2/9 + y^2/4 = 1 的椭圆，中心位于哪里？','位于原点。'),
             ('椭圆 x^2/9 + y^2/4 = 1 的长轴沿哪个方向？','沿 x 轴方向。'),
             ('椭圆 x^2/9 + y^2/4 = 1 的横向和纵向半轴长度分别是多少？','横向半轴为 3，纵向半轴为 2。')]}
]


def make_pdf(path, sample):
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    c = canvas.Canvas(str(path), pagesize=(595,842))
    c.setFillColorRGB(.06,.17,.20)
    c.setFont('STSong-Light', 22)
    c.drawString(45,775,sample['name'][:-4])
    c.setFillColorRGB(.35,.42,.43)
    c.setFont('STSong-Light',11)
    c.drawString(45,744,'QAproject 合成演示资料 / 不作为真实论文或观测数据')
    y = 700
    for paragraph in sample['evidence']:
        for start in range(0,len(paragraph),42):
            c.drawString(45,y,paragraph[start:start+42])
            y -= 21
        y -= 13
    left, bottom, width, height = 65,190,455,245
    c.setStrokeColorRGB(.83,.87,.85)
    low = min(0,min(sample['values']))
    high = max(sample['values']) * 1.2
    if high == low: high=low+1
    def py(v): return bottom + (v-low)/(high-low)*height
    c.setFont('Helvetica',10)
    for step in range(5):
        value=low+(high-low)*step/4
        y=py(value)
        c.line(left,y,left+width,y)
        c.drawRightString(left-8,y-3,f'{value:.2g}')
    for i,value in enumerate(sample['values']):
        x=left+35+i*105
        c.setFillColorRGB(.10,.48,.43)
        c.rect(x,min(py(0),py(value)),58,max(1,abs(py(value)-py(0))),fill=1,stroke=0)
        c.setFillColorRGB(.1,.2,.23)
        c.drawCentredString(x+29,py(value)+9,f'{value:g}')
        c.drawCentredString(x+29,bottom-22,sample['labels'][i])
    c.setFont('Helvetica',11)
    c.drawString(left,bottom+height+22,sample['unit'])
    c.setFont('STSong-Light',11)
    c.drawString(45,120,'图 1：上述记录的可视化。阅读时同时保留单位和示例范围。')
    c.setFont('Helvetica',9)
    c.drawString(45,45,'SYNTHETIC DEMO - QAproject / Page 1')
    c.save()


def seed_demo(store):
    folder = store.root / 'fixtures'
    folder.mkdir(exist_ok=True)
    for sample in SAMPLES:
        path = folder / sample['name']
        if not path.exists():
            make_pdf(path,sample)
        doc_id, created = store.import_file(sample['name'],path.read_bytes(),'demo')
        doc = store.document(doc_id)
        if created:
            units,pages = extract(store,doc,settings())
            store.update_document(doc_id,units=units,pages=pages,summary=sample['summary'],status='demo_complete',coverage='complete')
        else:
            units=doc['units']
        topic = store.topic(sample['topic'])
        unit=units[0]
        store.link_topic(doc_id,topic['id'],sample['keywords'],[unit['id']])
        reasons=[0,1,2] if '正弦' in sample['name'] else [0,1,1] if '椭圆' in sample['name'] else [0,0,2]
        for index,(question,answer) in enumerate(sample['qas']):
            row={'input':question,'thinking':sample['evidence'][reasons[index]],
                 'output':answer,'source':source_ref(doc['path'],unit['locator'],topic['name']),'image':unit['image'],'language':'zh'}
            cid=uid()
            with store.connect() as db:
                old=db.execute('SELECT id FROM candidates WHERE document_id=? AND fingerprint=?',(doc_id,normalized(question))).fetchone()
                if old:
                    db.execute('UPDATE candidates SET row_json=?,row_hash=? WHERE id=?',(encode(row),digest(row),old['id']))
                    db.execute('UPDATE qas SET row_json=? WHERE id=?',(encode(row),old['id']))
                    continue
                db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (cid,doc_id,unit['id'],topic['id'],encode(row),digest(row),unit['hash'],normalized(question),'approved',encode({'note':'预置演示问答；由本地合成资料构造，未调用 AI。','demo':True}),time.time()))
                db.execute('INSERT INTO qas VALUES(?,?,?)',(cid,encode(row),time.time()))
        with store.connect() as db:
            db.execute("UPDATE tasks SET status='succeeded',stage='demo_fixture' WHERE document_id=?",(doc_id,))
