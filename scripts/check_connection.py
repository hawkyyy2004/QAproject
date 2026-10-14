"""Small, explicit real-provider capability check; never prints credentials."""
from pathlib import Path
import argparse
import json
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from qaproject.glm import GLMClient
    from qaproject.settings import settings,api_key
    parser=argparse.ArgumentParser()
    parser.add_argument('--vision',action='store_true',help='验证实际图片输入，会进行一次额外调用')
    args=parser.parse_args()
    report={'key_configured':bool(api_key()),'text':False,'vision':False}
    if not report['key_configured']:
        print(json.dumps(report,ensure_ascii=False));return 1
    cfg=settings()
    client=GLMClient(cfg)
    try:
        result,stats=client.call('只返回 JSON 对象 {"ok":true}。',{'test':'本地接口连通检查'})
        report['text']=result.get('ok') is True
        report['text_stats']=stats
    except ValueError as e:
        report['text_error']=str(e)
    if args.vision:
        from reportlab.pdfgen import canvas
        import pymupdf
        folder=ROOT/'runtime/capability-check'
        folder.mkdir(parents=True,exist_ok=True)
        code=str(secrets.randbelow(900000)+100000)
        pdf=folder/'probe.pdf'
        c=canvas.Canvas(str(pdf),pagesize=(420,200))
        c.setFont('Helvetica-Bold',60)
        c.drawCentredString(210,80,code)
        c.save()
        image=folder/'probe.png'
        with pymupdf.open(pdf) as doc:
            doc[0].get_pixmap(dpi=144).save(image)
        try:
            result,stats=client.call('读取图片中六位数字，只返回 {"code":"实际数字"}，看不清返回空字符串。',
                {'task':'实际图像输入验证'},[{'id':'probe','image':str(image)}],force_model=cfg.get('vision_model') or cfg['text_model'])
            report['vision']=result.get('code')==code
            report['vision_stats']=stats
        except ValueError as e:
            report['vision_error']=str(e)
    out=ROOT/'runtime/capability-check'
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['text'] and (not args.vision or report['vision']) else 1


if __name__=='__main__':
    raise SystemExit(main())
