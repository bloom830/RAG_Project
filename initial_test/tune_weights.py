r"""
混合检索权重网格搜索脚本
遍历多组 (FAISS, BM25) 权重组合，跑 CRUD_RAG 评测，找出最佳权重分配。

用法：
    python tune_weights.py
    python tune_weights.py --samples 20
    python tune_weights.py --weights 0.3 0.5 0.7
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 复用 crud_eval 的函数
sys.path.insert(0, str(Path(__file__).parent))
from crud_eval import (
    load_crud_dataset, build_rag, evaluate,
    CRUD_DATA_PATH, RESULTS_DIR,
)


# ============= 默认配置 =============
DEFAULT_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_SAMPLES = 20


def grid_search(items: list[dict], faiss_weights: list[float], samples: int) -> list[dict]:
    """对每组权重跑一次评测，返回结果列表。"""
    results = []
    total = len(faiss_weights)

    for idx, faiss_w in enumerate(faiss_weights, 1):
        bm25_w = round(1.0 - faiss_w, 2)
        weights = (faiss_w, bm25_w)
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{total}] 测试权重 FAISS={faiss_w}, BM25={bm25_w}")
        print(f"{'=' * 70}")

        # 每组权重重新构建 RAG（embedding 有缓存，FAISS 重建快；BM25 构建也快）
        chain, retrieve_fn = build_rag(
            items,
            use_hybrid=True,
            use_cross_encoder=True,
            weights=weights,
        )

        report = evaluate(
            items, chain, retrieve_fn,
            use_hybrid=True,
            use_cross_encoder=True,
            limit=samples,
        )

        s = report["summary"]
        result = {
            "faiss_weight": faiss_w,
            "bm25_weight": bm25_w,
            "avg_keyword_overlap": s["avg_keyword_overlap"],
            "avg_number_hit_rate": s["avg_number_hit_rate"],
            "pass_rate": s["pass_rate"],
            "bad_case_count": s["bad_case_count"],
            "avg_response_time_sec": s["avg_response_time_sec"],
        }
        results.append(result)
        print(f"\n[结果] overlap={s['avg_keyword_overlap'] * 100:.1f}%  "
              f"num_hit={s['avg_number_hit_rate'] * 100:.1f}%  "
              f"pass={s['pass_rate']:.1f}%  "
              f"bad={s['bad_case_count']}  "
              f"time={s['avg_response_time_sec']:.2f}s")

    return results


def print_comparison_table(results: list[dict]):
    """打印权重对比表。"""
    print(f"\n{'=' * 80}")
    print("权重对比结果")
    print(f"{'=' * 80}")
    header = f"{'FAISS':>6} | {'BM25':>6} | {'overlap':>8} | {'num_hit':>8} | {'pass%':>6} | {'bad':>4} | {'time':>6}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['faiss_weight']:>6.1f} | {r['bm25_weight']:>6.1f} | "
              f"{r['avg_keyword_overlap'] * 100:>7.1f}% | "
              f"{r['avg_number_hit_rate'] * 100:>7.1f}% | "
              f"{r['pass_rate']:>5.1f}% | "
              f"{r['bad_case_count']:>4} | "
              f"{r['avg_response_time_sec']:>5.2f}s")
    print("-" * 80)


def find_best(results: list[dict]) -> dict:
    """找 overlap 最高的权重组合（平手时看 pass_rate）。"""
    return max(results, key=lambda x: (x["avg_keyword_overlap"], x["pass_rate"]))


def save_report(results: list[dict], best: dict):
    """保存调参报告。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "samples_per_weight": results[0].get("avg_response_time_sec") and len(results),
        "results": results,
        "best": best,
    }
    json_path = RESULTS_DIR / f"tune_weights_{timestamp}.json"
    txt_path = RESULTS_DIR / f"tune_weights_{timestamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # TXT 报告
    lines = []
    lines.append("=" * 80)
    lines.append(f"混合检索权重调参报告 ({timestamp})")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'FAISS':>6} | {'BM25':>6} | {'overlap':>8} | {'num_hit':>8} | {'pass%':>6} | {'bad':>4} | {'time':>6}")
    lines.append("-" * 80)
    for r in results:
        lines.append(f"{r['faiss_weight']:>6.1f} | {r['bm25_weight']:>6.1f} | "
                     f"{r['avg_keyword_overlap'] * 100:>7.1f}% | "
                     f"{r['avg_number_hit_rate'] * 100:>7.1f}% | "
                     f"{r['pass_rate']:>5.1f}% | "
                     f"{r['bad_case_count']:>4} | "
                     f"{r['avg_response_time_sec']:>5.2f}s")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"最优权重：FAISS={best['faiss_weight']}, BM25={best['bm25_weight']}")
    lines.append(f"  overlap={best['avg_keyword_overlap'] * 100:.1f}%  "
                 f"num_hit={best['avg_number_hit_rate'] * 100:.1f}%  "
                 f"pass={best['pass_rate']:.1f}%")
    lines.append("")
    lines.append(f"JSON: {json_path}")
    lines.append(f"TXT:  {txt_path}")

    text = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\n报告已保存：")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="混合检索权重网格搜索")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"每组权重评测样本数，默认 {DEFAULT_SAMPLES}")
    parser.add_argument("--weights", type=float, nargs="+",
                        default=DEFAULT_WEIGHTS,
                        help=f"FAISS 权重列表，默认 {DEFAULT_WEIGHTS}（BM25=1-FAISS）")
    args = parser.parse_args()

    # 环境检查
    import os
    if not os.getenv("ZHIPUAI_API_KEY"):
        print("[ERROR] 未配置 ZHIPUAI_API_KEY，请在 .env 中设置")
        sys.exit(1)
    if not CRUD_DATA_PATH.exists():
        print(f"[ERROR] 数据集不存在: {CRUD_DATA_PATH}")
        sys.exit(1)

    # 加载数据集
    items = load_crud_dataset(CRUD_DATA_PATH)
    print(f"[INFO] 加载数据集：共 {len(items)} 条有效记录")
    print(f"[INFO] 待测权重：{args.weights}")
    print(f"[INFO] 每组样本数：{args.samples}")
    print(f"[INFO] 预计总耗时：约 {len(args.weights) * args.samples * 15 / 60:.1f} 分钟")

    # 网格搜索
    results = grid_search(items, args.weights, args.samples)

    # 输出对比
    print_comparison_table(results)

    # 找最优
    best = find_best(results)
    print(f"\n最优权重（按 overlap，平手看 pass_rate）：")
    print(f"  FAISS={best['faiss_weight']}, BM25={best['bm25_weight']}")
    print(f"  overlap={best['avg_keyword_overlap'] * 100:.1f}%  "
          f"num_hit={best['avg_number_hit_rate'] * 100:.1f}%  "
          f"pass={best['pass_rate']:.1f}%  "
          f"bad={best['bad_case_count']}")

    # 保存报告
    save_report(results, best)


if __name__ == "__main__":
    main()
