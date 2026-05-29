# -*- coding: utf-8 -*-
"""量价关系量化交易系统 - 一键启动"""

import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    print('=' * 60)
    print('   量价关系量化交易系统  v1.0')
    print('   Volume-Price Signal Engine')
    print('=' * 60)
    print()
    print('  启动 Web 服务: http://localhost:5000')
    print('  按 Ctrl+C 停止')
    print()

    from web.app import app
    app.run(host='127.0.0.1', port=5000, debug=False)
