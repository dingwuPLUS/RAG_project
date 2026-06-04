# test_simple.py
import subprocess
import sys

print("1. 检查 Ollama 是否安装...")
try:
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    print(f"   ✅ {result.stdout.strip()}")
except:
    print("   ❌ Ollama 未找到，请安装 Ollama")
    sys.exit(1)

print("\n2. 检查模型是否存在...")
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "qwen2:7b" in result.stdout:
        print("   ✅ qwen2:7b 模型存在")
    else:
        print("   ❌ qwen2:7b 模型不存在，请运行: ollama pull qwen2:7b")
        sys.exit(1)
except:
    print("   ❌ 无法获取模型列表")

print("\n3. 测试 Ollama API...")
try:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2:7b", "prompt": "hi", "stream": False},
        timeout=30
    )
    print(f"   ✅ API 响应成功: {response.json().get('response', '')[:30]}...")
except Exception as e:
    print(f"   ❌ API 连接失败: {e}")
    print("\n   请确保:")
    print("   1. 运行 'ollama serve' 启动服务")
    print("   2. 服务窗口保持打开")

print("\n4. 测试 LangChain 连接...")
try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="qwen2:7b", temperature=0.7)
    response = llm.invoke("hi")
    print(f"   ✅ LangChain 连接成功: {response[:30]}...")
except Exception as e:
    print(f"   ❌ LangChain 连接失败: {e}")