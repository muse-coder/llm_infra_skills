import asyncio
import aiohttp
import time
import json
from typing import List, Dict

# ============================================================
# 公共前缀 Prompt（很长，用于触发框架的 Prefix Caching 优化）
# ============================================================
COMMON_PREFIX = """
你是一个专业的人工智能助手。以下是一段详细的人工智能领域知识背景介绍，请基于这些知识回答后续问题。

=== 人工智能背景知识 ===

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、
图像识别、自然语言处理和专家系统等。

人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，
将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，
但能像人那样思考、也可能超过人的智能。

机器学习（Machine Learning）是人工智能的核心，是使计算机具有智能的根本途径。机器学习主要研究
如何使计算机能够模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断
改善自身的性能。

深度学习（Deep Learning）是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的
算法。深度学习是机器学习中一种基于对数据进行表征学习的方法。观测值（例如一幅图像）可以使用多种
方式来表示，如每个像素强度值的向量，或者更抽象地表示成一系列边、特定形状的区域等。

自然语言处理（Natural Language Processing，简称NLP）是计算机科学领域与人工智能领域中的一个
重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门
融语言学、计算机科学、数学于一体的科学。

大型语言模型（Large Language Models，简称LLM）是一种基于深度学习的自然语言处理模型，它通过
在大量文本数据上进行训练来学习语言的统计规律。大型语言模型通常具有数十亿甚至数千亿个参数，能够
生成连贯、有意义的文本，并在各种自然语言处理任务上取得了显著的性能提升。

Transformer架构是目前大多数大型语言模型的基础架构，由Vaswani等人在2017年的论文
"Attention is All You Need"中提出。Transformer使用自注意力机制来处理序列数据，
相比传统的循环神经网络（RNN）和长短期记忆网络（LSTM），Transformer可以并行处理序列中的
所有元素，大大提高了训练效率。

强化学习（Reinforcement Learning，简称RL）是机器学习的一个重要分支，它研究智能体
（Agent）如何在环境中采取行动以最大化累积奖励。强化学习从人类和动物的学习方式中获得灵感，
通过试错的方式来学习最优策略。

计算机视觉（Computer Vision）是人工智能的一个重要应用领域，它研究如何使计算机能够理解和
处理图像和视频数据。计算机视觉技术已经广泛应用于人脸识别、目标检测、图像分类、语义分割等任务。

知识图谱（Knowledge Graph）是一种用于表示现实世界中实体及其关系的图结构数据库。知识图谱
将信息组织为节点（实体）和边（关系），使计算机能够理解和推理复杂的知识体系。

=== 基于以上背景知识，请回答以下问题 ===
"""

# ============================================================
# 不同的后缀问题（每个请求独有的部分）
# ============================================================
UNIQUE_QUESTIONS = [
    "AI的全称是什么？",
    "机器学习的核心目标是什么？",
    "深度学习与机器学习的关系是什么？",
    "NLP的全称是什么？",
    "Transformer架构是哪一年提出的？",
    "大型语言模型通常有多少参数？",
    "强化学习中智能体的目标是什么？",
    "计算机视觉有哪些主要应用？",
    "知识图谱使用什么数据结构？",
    "自注意力机制有什么优势？",
    "深度学习使用什么架构？",
    "LLM的全称是什么？",
]


def build_prompts(common_prefix: str, questions: List[str]) -> List[str]:
    """构建完整的 prompts，每个都包含公共前缀"""
    return [common_prefix + question for question in questions]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    request_id: int,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """发送单个异步请求"""
    payload = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": 50,
            "do_sample": False,
        },
    }

    async with semaphore:  # 控制并发数量
        start_time = time.time()
        try:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                result = await response.json()
                elapsed = time.time() - start_time

                print(f"[Request {request_id:02d}] ✅ 完成 | 耗时: {elapsed:.3f}s")
                print(f"[Request {request_id:02d}] 问题: {prompt[-30:]!r}")
                print(f"[Request {request_id:02d}] 回答: {result}")
                print("-" * 60)

                return {
                    "request_id": request_id,
                    "status": "success",
                    "elapsed": elapsed,
                    "result": result,
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Request {request_id:02d}] ❌ 失败 | 耗时: {elapsed:.3f}s | 错误: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "elapsed": elapsed,
                "error": str(e),
            }


async def run_batch_requests(
    url: str,
    prompts: List[str],
    max_concurrency: int = 4,
) -> List[Dict]:
    """
    批量并发发送请求
    
    Args:
        url: 服务地址
        prompts: prompt 列表
        max_concurrency: 最大并发数（控制服务器压力）
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, prompt, idx, semaphore)
            for idx, prompt in enumerate(prompts)
        ]

        print(f"\n🚀 开始发送 {len(tasks)} 个请求，最大并发数: {max_concurrency}")
        print(f"📝 公共前缀长度: {len(COMMON_PREFIX)} 字符")
        print("=" * 60)

        results = await asyncio.gather(*tasks)
        return list(results)


def print_summary(results: List[Dict], total_elapsed: float):
    """打印统计摘要"""
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    avg_time = sum(r["elapsed"] for r in success) / len(success) if success else 0
    min_time = min((r["elapsed"] for r in success), default=0)
    max_time = max((r["elapsed"] for r in success), default=0)

    print("\n" + "=" * 60)
    print("📊 批量请求统计摘要")
    print("=" * 60)
    print(f"  总请求数:     {len(results)}")
    print(f"  成功:         {len(success)}")
    print(f"  失败:         {len(failed)}")
    print(f"  总耗时:       {total_elapsed:.3f}s")
    print(f"  平均响应时间: {avg_time:.3f}s")
    print(f"  最快响应:     {min_time:.3f}s")
    print(f"  最慢响应:     {max_time:.3f}s")
    print(f"  吞吐量:       {len(success) / total_elapsed:.2f} req/s")
    print("=" * 60)


# ============================================================
# 多轮 Batch 发送（模拟真实场景）
# ============================================================
async def run_multi_batch(url: str, batch_size: int = 4, num_batches: int = 3):
    """
    多轮 batch 发送请求
    
    Args:
        url: 服务地址
        batch_size: 每批请求数量
        num_batches: 批次数量
    """
    all_prompts = build_prompts(COMMON_PREFIX, UNIQUE_QUESTIONS)

    # 如果问题不够，循环复用
    extended_prompts = []
    for i in range(num_batches * batch_size):
        extended_prompts.append(all_prompts[i % len(all_prompts)])

    total_start = time.time()
    all_results = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_prompts = extended_prompts[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"📦 Batch {batch_idx + 1}/{num_batches} | 包含 {len(batch_prompts)} 个请求")
        print(f"{'='*60}")

        batch_start_time = time.time()
        results = await run_batch_requests(url, batch_prompts, max_concurrency=batch_size)
        batch_elapsed = time.time() - batch_start_time

        all_results.extend(results)
        print(f"\n✅ Batch {batch_idx + 1} 完成，耗时: {batch_elapsed:.3f}s")

        # 批次间短暂等待
        if batch_idx < num_batches - 1:
            await asyncio.sleep(0.5)

    total_elapsed = time.time() - total_start
    print_summary(all_results, total_elapsed)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量 LLM 请求测试脚本（利用 Prefix Cache）")
    parser.add_argument("--url", default="http://localhost:8066", help="服务地址")
    parser.add_argument("--batch-size", type=int, default=4, help="每批请求数量")
    parser.add_argument("--num-batches", type=int, default=3, help="批次数量")
    parser.add_argument("--concurrency", type=int, default=4, help="最大并发数")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          LLM 批量请求测试 - Prefix Cache 优化            ║
╠══════════════════════════════════════════════════════════╣
║  服务地址:   {args.url:<44} ║
║  每批大小:   {args.batch_size:<44} ║
║  批次数量:   {args.num_batches:<44} ║
║  最大并发:   {args.concurrency:<44} ║
║  前缀长度:   {len(COMMON_PREFIX):<44} ║
╚══════════════════════════════════════════════════════════╝
    """)

    asyncio.run(
        run_multi_batch(
            url=args.url,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
        )
    )
