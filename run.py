"""
启动脚本
运行Flask Web应用
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.api.app import app, logger
from config import FLASK_HOST, FLASK_PORT, DEBUG

if __name__ == '__main__':
    print("="*60)
    print("Yelp评论分析系统 - Web应用")
    print("="*60)
    print(f"\n启动服务器...")
    print(f"访问地址: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"调试模式: {DEBUG}")
    print("\n按 Ctrl+C 停止服务器\n")
    print("="*60)

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
