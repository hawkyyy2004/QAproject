"""Reference annotation interchange; no QA generation or candidate writes."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
import hashlib
from urllib.parse import urlsplit

from .store import encode,digest


def init_schema(store):
    with store.connect() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS corpus_reference_records(
          annotation_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, row_json TEXT NOT NULL,
          row_hash TEXT NOT NULL)''')


def reference_files(store,version,image_prefix='images'):
    from .instructions import list_annotations
    annotations=list_annotations(store,version['id'])
    init_schema(store)
    with store.connect() as db:
        records=db.execute('SELECT * FROM corpus_reference_records WHERE version_id=?',(version['id'],)).fetchall()
    references={}
    for record in records:
        row=json.loads(record['row_json'])
        if digest(row)!=record['row_hash']:raise ValueError('原始参考标注绑定失效')
        references[record['annotation_id']]=row
    units={u['id']:u for u in version['units'] if u.get('image')}
    bio=[];sonar=[];files={};warnings=[]
    for u in units.values():
        image_path=image_prefix+'/'+Path(u['image']).name
        files[image_path]=Path(u['image']).read_bytes()
        expected=version['manifest']['files']['images/'+Path(u['image']).name]
        if hashlib.sha256(files[image_path]).hexdigest()!=expected:raise ValueError('导出期间标注图片已变化')
        approved=[a for a in annotations if a['unit_id']==u['id'] and a['verified']]
        for a in approved:
            if a['kind']=='label':
                bio.append(dict(image_path=image_path,category=a['value'],prompt=references.get(a['id'],{}).get('prompt','')))
            if a['kind']=='bbox':
                value=a['value'];box=value['coords']
                if value['coordinate_system']=='pixel':
                    box=[box[0]/value['width'],box[1]/value['height'],box[2]/value['width'],box[3]/value['height']]
                if not value.get('label'):
                    warnings.append('存在未标明类别的目标框，仅保留内部审计，不冒充完整声呐标注。');continue
                ref=references.get(a['id'],{})
                sonar.append(dict(image_path=image_path,object=value['label'],bnd_box=box,input=ref.get('input',''),output=ref.get('output','')))
    files['Open-Dataset/annotations.json']=encode(annotations).encode('utf-8')
    files['Open-Dataset/OceanVision/OceanVision_train_data.json']=encode(bio).encode('utf-8')
    files['Open-Dataset/OceanSona/sona_train_data.json']=encode(sonar).encode('utf-8')
    files['Open-Dataset/README.txt']=('图片路径从资料包根目录解析。仅人工确认的类别/框进入参考文件；未确认标注保留于annotations.json。\n'
        '空prompt/input/output表示尚未加工指令，不是已审核QA或可直接训练的数据。\n'+'\n'.join(warnings)).encode('utf-8')
    return files


def import_reference(store,version_id,records,source_note,verified=False,class_map=None):
    from . import corpus,instructions
    version=corpus.verify(store,version_id)
    init_schema(store)
    units=[u for u in version['units'] if u.get('image')]
    doc=corpus._document(store,version['document_id'])
    results=[]
    for index,row in enumerate(records):
        try:
            if not isinstance(row,dict):raise ValueError('每条标注必须为对象')
            name=row.get('image_path','')
            if not isinstance(name,str) or not name or '\\' in name or PureWindowsPath(name).drive or urlsplit(name).scheme or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('image_path必须为安全相对路径')
            # Match only actual registered images; never open an arbitrary imported path.
            matches=[]
            for u in units:
                aliases={'images/'+Path(u['image']).name,'Open-Dataset/images/'+Path(u['image']).name}
                if version['kind']=='image':aliases.add(doc['name'])
                if name in aliases or (version['kind']=='image' and PurePosixPath(name).name==doc['name']):matches.append(u)
            if len(matches)!=1:raise ValueError('图片未登记或映射不唯一，请按资料包图片路径填写')
            unit=matches[0]
            label=row.get('category') or row.get('object')
            if not isinstance(label,str) or not label.strip():raise ValueError('缺少category或object类别')
            original_label=label
            label=(class_map or {}).get(label,label)
            if not isinstance(label,str) or not label.strip():raise ValueError('类别映射必须为非空文字')
            for key in ('prompt','input','output'):
                if key in row and (not isinstance(row[key],str) or len(row[key])>20000):raise ValueError('参考指令字段必须为不超过20000字符的文字')
            if 'bnd_box' in row:
                kind='bbox';value=dict(box=row['bnd_box'],coordinates='normalized',label=label)
            else:kind='label';value=dict(label=label)
            note=source_note+(f' | 原类别：{original_label}' if original_label!=label else '')
            saved=instructions.add_annotation(store,version_id,unit['id'],kind,value,verified,note)
            with store.connect() as db:
                existing=db.execute('SELECT row_hash FROM corpus_reference_records WHERE annotation_id=?',(saved['id'],)).fetchone()
                if existing and existing[0]!=digest(row):raise ValueError('相同标注已有不同参考文本，未覆盖历史记录')
                db.execute('INSERT OR IGNORE INTO corpus_reference_records VALUES(?,?,?,?)',(saved['id'],version_id,encode(row),digest(row)))
            results.append(dict(index=index,id=saved['id']))
        except (ValueError,TypeError,KeyError) as error:
            results.append(dict(index=index,error=str(error)))
    return dict(results=results,imported=sum('id' in r for r in results),failed=sum('error' in r for r in results))
