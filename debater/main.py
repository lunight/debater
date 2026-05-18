"""CLI 入口

提供命令行方式运行多模型辩论。
"""

import argparse
import json
from pathlib import Path

from .graph import run_debate


def main():
    parser = argparse.ArgumentParser(description="多模型辩论系统")
    parser.add_argument(
        "question",
        type=str,
        help="要分析的问题",
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default="",
        help="补充背景资料（如财务数据、新闻等）",
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="question_analysis",
        choices=["question_analysis", "financial_interpretation", "event_analysis", "risk_assessment"],
        help="分析场景类型",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="最大辩论轮次（默认: 3）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出结果到 JSON 文件",
    )
    
    args = parser.parse_args()
    
    # 运行辩论
    result = run_debate(
        question=args.question,
        context=args.context,
        scenario_type=args.scenario,
        max_rounds=args.rounds,
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📋 最终结论")
    print("=" * 60)
    print(result.get("final_answer", "（无最终结论）"))
    
    if result.get("divergence_points"):
        print("\n⚠️  分歧点:")
        for dp in result["divergence_points"]:
            print(f"   - {dp}")
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "question": result["question"],
            "scenario_type": result["scenario_type"],
            "rounds": result["round"],
            "final_answer": result["final_answer"],
            "confidences": result["confidences"],
            "divergence_points": result.get("divergence_points", []),
            "reasoning_process": result.get("reasoning_process", ""),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
