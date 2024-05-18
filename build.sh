
#!/bin/bash
set -x
# 检查是否至少提供了一个参数
if [ "$#" -ne 1 ]; then
    echo "使用方法: $0 参数"
    exit 1
fi

# 如果一切正常，这里可以访问你的参数，$1 表示第一个参数
echo "你提供的参数是: $1"

docker build --platform=linux/amd64 -t allenhaozi/gradio-chat:$1 .

docker push allenhaozi/gradio-chat:$1
