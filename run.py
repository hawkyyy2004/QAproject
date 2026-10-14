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
    args=parser.parse_args()
    if not args.no_browser:
        threading.Timer(2,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    print(f'QAproject: http://127.0.0.1:{args.port} / key 不输出，关闭服务请按 Ctrl+C。')
    uvicorn.run(create_app(),host='127.0.0.1',port=args.port,log_level='warning')
