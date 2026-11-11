"""Bounded webpage capture. Pin public DNS addresses on every redirect hop."""
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin, urlunsplit

MAX_BYTES = 8 * 1024 * 1024
_CAPTURES = threading.BoundedSemaphore(2)
_DNS_SLOTS = threading.BoundedSemaphore(2)
_DNS = ThreadPoolExecutor(max_workers=2,thread_name_prefix='web-dns')


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self.stack = [], [], []
        self.ignored = 0
        self.in_title = False
        self.in_head = False
        self.svg_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'nav', 'footer', 'noscript', 'template'}:
            self.ignored += 1
            self.stack.append(tag)
        if tag == 'head': self.in_head = True
        if tag == 'svg': self.svg_depth += 1
        if tag == 'title' and not self.svg_depth and not self.title: self.in_title = True
        if tag in {'td','th'} and not self.ignored: self.parts.append(' | ')
        if tag in {'p','div','br','li','h1','h2','h3','h4','tr','section','article'} and not self.ignored:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if self.stack and tag == self.stack[-1]:
            self.stack.pop(); self.ignored -= 1
        if tag == 'title': self.in_title = False
        if tag == 'head': self.in_head = False
        if tag == 'svg': self.svg_depth = max(0,self.svg_depth-1)
        if tag in {'p','div','li','h1','h2','h3','tr','section','article'} and not self.ignored:
            self.parts.append('\n')

    def handle_data(self, text):
        if self.in_title: self.title.append(text)
        elif not self.ignored: self.parts.append(text)


def decode_html(raw, charset=None):
    if isinstance(raw, bytes):
        declared = re.search(br'<meta\b[^>]*\bcharset\s*=\s*["\']?([\w-]+)', raw[:4096], re.I)
        charset = ('utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else charset or
                   (declared.group(1).decode('ascii', errors='ignore') if declared else 'utf-8'))
        try: text = raw.decode(charset, errors='replace')
        except LookupError: text = raw.decode('utf-8', errors='replace')
    else: text = raw
    return text


def extract_html(raw, charset=None):
    text=decode_html(raw,charset)
    parser = ArticleParser(); parser.feed(text); parser.close()
    lines = [re.sub(r'[ \t\r\f\v]+', ' ', line).strip() for line in ''.join(parser.parts).splitlines()]
    return dict(title=' '.join(''.join(parser.title).split())[:500],
                text='\n\n'.join(line for line in lines if line))


def public_target(url):
    p = urlsplit(url)
    if p.scheme not in {'http','https'} or not p.hostname or p.username or p.password:
        raise ValueError('请输入有效的 HTTP/HTTPS 公网网页地址，不接受账号密码。')
    try: port = p.port or (443 if p.scheme == 'https' else 80)
    except ValueError: raise ValueError('网页端口无效。') from None
    if port not in {80,443}: raise ValueError('网页接入仅支持标准 HTTP/HTTPS 端口。')
    hostname = p.hostname.encode('idna').decode('ascii')
    if not _DNS_SLOTS.acquire(blocking=False): raise ValueError('网页域名查询繁忙，请稍后重试。')
    future = _DNS.submit(socket.getaddrinfo,hostname,port,type=socket.SOCK_STREAM)
    future.add_done_callback(lambda _: _DNS_SLOTS.release())
    try: addresses = list(dict.fromkeys(x[4][0] for x in future.result(timeout=5)))
    except TimeoutError: raise ValueError('网页域名查询超时，请稍后重试。') from None
    except OSError: raise ValueError('网页域名无法解析，请检查网络或改为导入已保存的 HTML。') from None
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError('网页地址不能指向本机、内网或保留地址。')
    normalized = urlunsplit((p.scheme, p.netloc, p.path or '/', p.query, ''))
    return normalized, hostname, port, addresses[0]


def capture(url):
    if not _CAPTURES.acquire(blocking=False): raise ValueError('正在接入其他网页，请稍后重试。')
    try: return _capture(url)
    finally: _CAPTURES.release()


def capture_image(url):
    if not _CAPTURES.acquire(blocking=False):raise ValueError('图片采集繁忙，请稍后重试。')
    try:return _capture(url,image=True)
    finally:_CAPTURES.release()


def _capture(url, image=False):
    original = url
    deadline = time.monotonic()+35
    for _ in range(6):
        url, host, port, address = public_target(url)
        p = urlsplit(url)
        conn = http.client.HTTPConnection(host, port, timeout=20)
        timer = None
        try:
            remaining = deadline-time.monotonic()
            if remaining<=0: raise ValueError('网页采集超过总时限，未保存不完整内容。')
            # HTTPConnection otherwise resolves a second time; use the checked IP.
            conn.sock = socket.create_connection((address, port), timeout=min(remaining,15))
            if p.scheme == 'https':
                conn.sock = ssl.create_default_context().wrap_socket(conn.sock, server_hostname=host)
            sock = conn.sock
            def abort():
                try: sock.shutdown(socket.SHUT_RDWR)
                except (OSError,AttributeError): pass
                try: sock.close()
                except (OSError,AttributeError): pass
            timer=threading.Timer(max(.01,deadline-time.monotonic()),abort)
            timer.daemon=True;timer.start()
            conn.request('GET', (p.path or '/') + ('?' + p.query if p.query else ''),
                         headers={'User-Agent':'OceanPileQA/1.0 (research corpus capture)',
                                  'Accept':'image/png,image/jpeg,image/webp' if image else 'text/html,text/plain;q=0.9', 'Accept-Encoding':'identity'})
            response = conn.getresponse()
            if response.status in {301,302,303,307,308}:
                location = response.getheader('location')
                if not location: raise ValueError('网页跳转没有提供目标地址。')
                url = urljoin(url, location)
                continue
            if response.status != 200: raise ValueError(f'网页读取失败（HTTP {response.status}），可导入已保存的 HTML。')
            content_type = response.getheader('content-type','').lower()
            if image and content_type.split(';')[0] not in {'image/png','image/jpeg','image/webp'}:
                raise ValueError('网页图片仅接收真实PNG/JPEG/WebP，不接收SVG或动态图片。')
            if not image and not any(t in content_type for t in ('text/html','application/xhtml+xml','text/plain')):
                raise ValueError('该地址不是可读取的 HTML/文字网页，请下载资料后导入。')
            if response.getheader('content-encoding','identity').lower() not in {'','identity'}:
                raise ValueError('网页返回不支持的压缩编码，请导入已保存的 HTML。')
            maximum=5*1024*1024 if image else MAX_BYTES
            raw = response.read(maximum+1)
            if time.monotonic()>=deadline: raise ValueError('网页采集超过总时限，未保存不完整内容。')
            if len(raw)>maximum: raise ValueError('图片超过5MB限额。' if image else '网页超过 8MB 接入限额，未截断保存。')
            if image:
                from PIL import Image
                import io
                with Image.open(io.BytesIO(raw)) as picture:
                    if picture.format not in {'PNG','JPEG','WEBP'} or picture.width*picture.height>40000000:
                        raise ValueError('网页图片格式或像素数不符合限制。')
                    width,height=picture.size;fmt=picture.format;picture.verify()
                return dict(raw=raw,url=url,width=width,height=height,
                            suffix={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
            charset = re.search(r'charset=([^;\s]+)', content_type)
            parsed = extract_html(raw, charset.group(1).strip('"\'') if charset else None) if 'html' in content_type else {
                'title':host, 'text':raw.decode('utf-8', errors='replace')}
            if re.search(r'page\s+not\s+found|error\s*404|404\s*not\s*found|页面不存在|页面未找到',parsed['title'],re.I):
                raise ValueError('该网址返回错误页面，未登记为知识资料，请核对原始地址。')
            if not parsed['text'].strip(): raise ValueError('网页没有可读正文；动态网页请导入完整保存的页面。')
            return dict(parsed, raw=raw, url=url, requested_url=original,
                        charset=charset.group(1).strip('"\'') if charset else None,
                        suffix='.html' if 'html' in content_type else '.txt',
                        content_hash=hashlib.sha256(raw).hexdigest())
        except (OSError, http.client.HTTPException) as error:
            raise ValueError(f'网页连接失败（{type(error).__name__}），请检查网络或导入已保存的 HTML。') from None
        finally:
            if timer: timer.cancel()
            conn.close()
    raise ValueError('网页跳转次数过多，未采集。')
