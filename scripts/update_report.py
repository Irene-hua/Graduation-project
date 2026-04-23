import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SUMMARY = BASE / "summary.json"
REPORT = BASE / "report.md"

def find_metric(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            res = find_metric(v, names)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for v in obj:
            res = find_metric(v, names)
            if res is not None:
                return res
    return None

def fmt(v):
    # 保留 4 位小数（如果值 <= 1 则显示小数形式）
    try:
        v = float(v)
    except Exception:
        return "0.0000"
    return f"{v:.4f}"

def main():
    if not SUMMARY.exists():
        print(f"summary.json not found: {SUMMARY}")
        return
    if not REPORT.exists():
        print(f"report.md not found: {REPORT}")
        return

    with SUMMARY.open("r", encoding="utf-8") as f:
        data = json.load(f)

    p = find_metric(data, {"precision", "prec", "p", "precision_score"})
    r = find_metric(data, {"recall", "rec", "r", "recall_score"})
    f1 = find_metric(data, {"f1", "f1-score", "f1_score", "fscore", "f1score"})

    # 如果没找到，尝试查找 overall 或 macro/micro 等常见嵌套
    #（find_metric 已会递归搜索，所以通常能找到）

    p_s = fmt(p) if p is not None else "0.0000"
    r_s = fmt(r) if r is not None else "0.0000"
    f1_s = fmt(f1) if f1 is not None else "0.0000"

    new_block = "\n".join([
        "<!--METRICS_START-->",
        f"Precision: {p_s}",
        f"Recall: {r_s}",
        f"F1: {f1_s}",
        "<!--METRICS_END-->"
    ])

    text = REPORT.read_text(encoding="utf-8")
    if "<!--METRICS_START-->" in text and "<!--METRICS_END-->" in text:
        text = re.sub(
            r"<!--METRICS_START-->.*?<!--METRICS_END-->",
            new_block,
            text,
            flags=re.S
        )
    else:
        # 如果没有标记，则在文件末尾追加
        text = text.rstrip() + "\n\n" + new_block + "\n"

    REPORT.write_text(text, encoding="utf-8")
    print(f"Updated report.md with Precision={p_s}, Recall={r_s}, F1={f1_s}")

if __name__ == "__main__":
    main()

