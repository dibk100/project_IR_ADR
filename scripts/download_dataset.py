"""Download datasets for phase01 difficulty verification.

GSM8K (필수)와 MATH-500 (optional)을 HuggingFace Hub에서 받아
로컬 data/ 디렉토리에 jsonl 형태로 저장한다.
 + TIGER-Lab/MMLU-Pro도 다운

이후 run_rollouts.py 등이 오프라인에서도 동작할 수 있도록
HF cache에도 함께 남는다.

Usage:
    python download_dataset.py                # GSM8K + MATH-500 모두 다운로드
    python download_dataset.py --only gsm8k   # GSM8K만
    python download_dataset.py --data-dir ./data
"""

import os
import json
import argparse

from datasets import load_dataset

# (출력 파일명, dataset id, config, split)
SOURCES = {
    "gsm8k": {
        "path": "gsm8k",
        "config": "main",           # main, socratic
        "split": "test",
        "out": "gsm8k_main_test.jsonl",
    },
    "mmlu_pro": {
        "path": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "split": "test",
        "out": "mmlu_pro_test.jsonl",
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "out": "math500_test.jsonl",
    },
}


def dump_jsonl(dataset, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def download_one(name: str, data_dir: str):
    spec = SOURCES[name]
    print(f"[{name}] downloading {spec['path']} "
          f"(config={spec['config']}, split={spec['split']}) ...")

    if spec["config"]:
        dataset = load_dataset(spec["path"], spec["config"], split=spec["split"])
    else:
        dataset = load_dataset(spec["path"], split=spec["split"])

    out_path = os.path.join(data_dir, spec["out"])
    dump_jsonl(dataset, out_path)

    print(f"[{name}] saved {len(dataset)} examples -> {out_path}")
    print(f"[{name}] columns: {dataset.column_names}")


def main():
    parser = argparse.ArgumentParser(description="Download phase01 datasets")
    parser.add_argument(
        "--only",
        choices=list(SOURCES.keys()),
        default=None,
        help="특정 데이터셋만 다운로드 (기본: 전체)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="저장할 디렉토리 (default: ./data)",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    targets = [args.only] if args.only else list(SOURCES.keys())
    for name in targets:
        try:
            download_one(name, args.data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
