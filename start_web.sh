#!/bin/bash
# Fund Trade Agent Web UI 启动脚本
# 使用方法: 双击运行 或 终端执行 bash start_web.sh

cd "$(dirname "$0")"
source venv/bin/activate

# 检查是否已在运行
if lsof -i :5001 > /dev/null 2>&1; then
    echo "⚠️  端口 5001 已被占用，Trade Agent 可能已在运行"
    echo "   如需重启，先运行: kill \$(lsof -t -i :5001)"
    echo ""
    echo "🌐 直接访问: http://localhost:5001"
    open http://localhost:5001
    exit 0
fi

echo "=================================================="
echo "  Fund Trade Agent Web UI"
echo "=================================================="
echo ""
echo "  🌐 访问地址: http://localhost:5001"
echo "  📋 按 Ctrl+C 停止服务"
echo ""

# 自动打开浏览器
open http://localhost:5001 &

python3 web_app.py
