from chunking_experiment import *
import glob

# 1. 读取一个示例文档（用你 docs 里的任意 txt）
sample_file = "docs/奥特曼排行榜.txt"  # 改成你的文件
if not os.path.exists(sample_file):
    # 如果没有，创建一个示例文本
    sample_text = """奥特曼排行榜

迪迦奥特曼：平成三部曲之首，最受欢迎的奥特曼。
赛罗奥特曼：赛文之子，实力强大，形态众多。
欧布奥特曼：融合形态，使用卡片变身。
泽塔奥特曼：令和时代的代表，获得星云赏。
捷德奥特曼：贝利亚之子，身世复杂但正义。
"""
    os.makedirs("docs", exist_ok=True)
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

# 2. 加载原始文本
with open(sample_file, "r", encoding="utf-8") as f:
    original_text = f.read()
print(f"原始文档长度: {len(original_text)} 字符")
print(original_text)
print("\n" + "="*60 + "\n")

# 3. 定义要测试的策略
strategies = {
    "递归分割 (500, 50)": lambda t: recursive_split(t, 500, 50),
    "递归分割 (200, 20)": lambda t: recursive_split(t, 200, 20),
    "固定字符分割 (500, 50)": lambda t: fixed_char_split(t, 500, 50),
    "Token 分割 (100 tokens)": lambda t: token_split(t, 100, 10),
    "NLTK 句子分割 (每2句一块)": lambda t: nltk_sentence_split(t, 2),
}

# 4. 运行每个策略并打印结果
for name, split_func in strategies.items():
    print(f"\n🔪 策略: {name}")
    try:
        chunks = split_func(original_text)
        print(f"  生成块数: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # 只展示前3块
            print(f"  块 {i+1} (长度 {len(chunk)}): {chunk[:80]}...")
    except Exception as e:
        print(f"  出错: {e}")
    print("-"*50)