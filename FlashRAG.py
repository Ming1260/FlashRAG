import sys
import os

# ==============================
# 将本地 flashrag 库添加到 Python 路径
# ==============================
flashrag_parent_dir = "/root/FlashRAG"
if os.path.isdir(flashrag_parent_dir):
    sys.path.insert(0, flashrag_parent_dir)
    print(f"✅ 已将 {flashrag_parent_dir} 添加到 Python 路径")
else:
    print(f"⚠️ 警告：路径 {flashrag_parent_dir} 不存在，请检查！")

import json
from flashrag.config import Config
from flashrag.generator import HFCausalLMGenerator

# ==============================
# 自定义生成器类：强制生成简洁答案
# ==============================
class ConciseRAGGenerator(HFCausalLMGenerator):
    def _build_prompt(self, query, retrieved_docs):
        # 构建极简 prompt：只给文档 + 问题，要求直接答
        context = "\n".join([doc['contents'] for doc in retrieved_docs])
        # 关键：明确指令 + 禁止解释
        prompt = (
            f"根据以下资料，用最简短的方式回答问题，不要解释，不要重复问题，只输出答案本身。\n\n"
            f"资料：{context}\n\n"
            f"问题：{query}\n\n"
            f"答案："
        )
        return prompt

# ==============================
# 配置区
# ==============================

# 知识库路径
CORPUS_PATH = "/public/modelscope-datasets/hhjinjiajie/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl"
INDEX_PATH = "/public/modelscope-datasets/hhjinjiajie/FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5.index"

# 模型配置
QWEN_MODEL = "/public/huggingface-models/Qwen/Qwen2.5-32B"

# flashrag 配置
config_dict = {
    "generator_model": QWEN_MODEL,
    "use_gpu": True,
    "gpu_device": 0,
    "dtype": "float16",
    "retrieval_method": "e5",
    "index_path": INDEX_PATH,
    "corpus_path": CORPUS_PATH,
    "retrieval_topk": 3,  # 减少检索文档数量，避免信息冗余
}

# 初始化配置
config = Config(config_dict=config_dict)

# 初始化生成器（使用自定义类）
generator = ConciseRAGGenerator(config)

# ==============================
# 工具函数
# ==============================
def clean_answer(text: str) -> str:
    """只保留实质内容，提取最核心答案"""
    if not text or text.lower() in ["", "unknown", "none", "null"]:
        return "未知"
    
    import re
    # 去除模板前缀
    text = re.sub(
        r"^(答案是[:：]?\s*|The answer is\s*|根据.*?[，,]?\s*|参考.*?[：:]?\s*|综上所述[，,]?\s*|答案[:：]?\s*)",
        "",
        text,
        flags=re.IGNORECASE
    )
    
    # 只保留第一句话（以句号、问号、感叹号、换行或分号结束）
    first_sentence = re.split(r'[。！？；\n]', text.strip(), maxsplit=1)[0].strip()
    
    # 去除引号和多余空格
    first_sentence = first_sentence.strip('"“”\'')
    
    return first_sentence if first_sentence else "未知"

# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    # 假设 json文件存在
    try:
        with open('data_b.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("❌ 错误: 文件未找到。请确保文件存在于当前目录。")
        exit()
    
    results = []
    for q in questions:
        print(f"处理 ID: {q['id']}")
        
        try:
            # 直接将问题传给 generator，它会内部完成 RAG 流程
            raw_answer = generator.generate(
                q['input_field'], # 直接传入问题
                max_new_tokens=96,  # 缩短生成长度，避免啰嗦
                temperature=0.0,    # 降低随机性，提高确定性
                do_sample=False     # 确定性生成
            )[0]
            
            final_answer = clean_answer(raw_answer)
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            final_answer = "未知"
        
        results.append({
            "id": q["id"],
            "output_field": final_answer
        })
    
    with open('result_b.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print("\n✅ 处理完成！结果已保存至 result_b.jsonl")