import socket
import pytest

from qaproject.web_sources import extract_html, public_target, capture


def test_html_keeps_real_body_and_excludes_navigation():
    data = extract_html('<title>海洋资料</title><nav>登录</nav><article><h1>盐度</h1>'
                        '<p>海水含有溶解盐。</p><script>ignore()</script><p>范围需明确。</p></article>')
    assert data['title'] == '海洋资料'
    assert '海水含有溶解盐。' in data['text']
    assert '登录' not in data['text'] and 'ignore' not in data['text']


def test_table_cells_and_article_conditions_are_not_lost():
    data=extract_html('<article><header>仅适用于盐度大于30的海水</header><p>该方法有效。</p>'
                      '<table><tr><td>1</td><td>2</td></tr></table></article>')
    assert '盐度大于30' in data['text']
    assert '1 | 2' in data['text'] and '12' not in data['text']


def test_body_charset_example_does_not_change_encoding():
    assert '海水' in extract_html('<p>海水 charset=latin-1</p>'.encode())['text']


def test_svg_titles_do_not_pollute_article_title():
    result=extract_html('<head><title>海洋资料</title></head><svg><title>icon</title></svg><p>正文。</p>')
    assert result['title']=='海洋资料'


def test_soft_404_is_rejected_before_registration(monkeypatch):
    import qaproject.web_sources as web
    monkeypatch.setattr(web,'public_target',lambda url:(url,'example.org',80,'8.8.8.8'))
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:object())
    class Connection:
        status=200
        def __init__(self,*a,**k):pass
        def request(self,*a,**k):pass
        def getresponse(self):return self
        def close(self):pass
        def getheader(self,key,default=None):return 'text/html' if key=='content-type' else default
        def read(self,*a):return b'<head><title>Page Not Found: Error 404</title></head><p>Try search.</p>'
    monkeypatch.setattr(web.http.client,'HTTPConnection',Connection)
    with pytest.raises(ValueError,match='错误页面'):capture('http://example.org/missing')


@pytest.mark.parametrize('url',['file:///etc/passwd','http://a:b@example.com',
                                'http://example.com:8000','http://localhost'])
def test_invalid_or_private_urls_rejected(url,monkeypatch):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('127.0.0.1',80))])
    with pytest.raises(ValueError): public_target(url)


def test_any_private_dns_result_rejected(monkeypatch):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[
        (2,1,6,'',('8.8.8.8',443)),(2,1,6,'',('10.0.0.1',443))])
    with pytest.raises(ValueError): public_target('https://example.com')


def test_redirect_to_private_address_is_not_followed(monkeypatch):
    import qaproject.web_sources as web
    monkeypatch.setattr(socket,'getaddrinfo',lambda host,*a,**k:[
        (2,1,6,'',('8.8.8.8' if host=='example.com' else '127.0.0.1',80))])
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:object())
    class Connection:
        def __init__(self,*a,**k):pass
        def request(self,*a,**k):pass
        def getresponse(self):return self
        def close(self):pass
        status=302
        def getheader(self,key,*a):return 'http://localhost/secret'
    monkeypatch.setattr(web.http.client,'HTTPConnection',Connection)
    with pytest.raises(ValueError,match='内网'): capture('http://example.com')


def test_html_parser_route_is_text_only(tmp_path):
    from qaproject.store import Store
    from qaproject.parsers import extract
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('ocean.html',b'<title>Sea</title><p>Ocean salinity varies.</p>','live')
    units,pages=extract(store,store.document(doc_id),{'pdf_layout':'off'})
    assert pages==0 and units[0]['text']=='Ocean salinity varies.'
    assert units[0]['locator']=='seg-01' and not units[0]['image']
