"""Only controlled layout diagnostics may cross the subprocess boundary."""
import json
from subprocess import CompletedProcess

import pytest

from qaproject import layout
from qaproject.store import Store
from test_layout_manifest import graphics_pdf


@pytest.mark.parametrize('warning,expected',[
    ('第9页未识别出证据，未当作空白页跳过','第9页'),
    ('第3页独立图块无法保存，未丢弃图片或改用整页截图','第3页'),
    ('MinerU页面不完整，未截断或改用整页截图','页面不完整'),
    ('MinerU 只识别出 100 字符，PDF 文字层有 1000 字符，本次版面未通过完整性校验','识别100字符'),
])
def test_known_rejection_reason_reaches_caller(tmp_path,monkeypatch,warning,expected):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live')
    monkeypatch.setattr(layout,'runner',lambda config:'isolated-mineru')
    output=json.dumps({'status':'rejected','warnings':[warning]},ensure_ascii=False)
    monkeypatch.setattr(layout.subprocess,'run',lambda *a,**kw:CompletedProcess([],1,output,'untrusted child stderr'))
    pages,message=layout.prepare(store,store.document(doc),{'require_mineru':True})
    assert not pages and expected in message and 'untrusted' not in message


def test_arbitrary_child_error_is_not_echoed():
    marker='untrusted-diagnostic-marker'
    for output in [marker,json.dumps({'status':'failed','error':marker}),
                   json.dumps({'status':'rejected','warnings':[marker]}),
                   json.dumps({'status':'rejected','warnings':marker})]:
        assert marker not in layout.preparation_error(output)
