from pathlib import Path
import argparse
import sys
import threading
import webbrowser

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    import uvicorn
    from qaproject.app import create_app
    parser=argparse.ArgumentParser(description='启动 QAproject 本地初版')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--data-dir',type=Path,help='持久数据目录，默认沿用项目 runtime')
    parser.add_argument('--library-root',type=Path,help='可选服务器资料目录，不配置时仍使用本机上传')
    parser.add_argument('--host',default='127.0.0.1',help='监听地址，默认仅本机；远程访问还需配置 allowed_hosts')
    args=parser.parse_args()
    if not args.no_browser:
        threading.Timer(2,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    print(f'QAproject: http://127.0.0.1:{args.port} / key 不输出，关闭服务请按 Ctrl+C。')
    uvicorn.run(create_app(runtime=args.data_dir,library_root=args.library_root),host=args.host,port=args.port,log_level='warning')
