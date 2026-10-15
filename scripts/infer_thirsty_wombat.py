"""用微调后的 RF-DETR Nano 对单张图做检测。

用法（test-gpu 环境）:
    python scripts/infer_thirsty_wombat.py --image path\\to\\image.jpg
    python scripts/infer_thirsty_wombat.py --image path\\to\\image.jpg --checkpoint output\\thirsty-wombat2\\checkpoint_best_total.pth
"""

import argparse
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "output" / "thirsty-wombat2" / "checkpoint_best_total.pth"
DEFAULT_OUTPUT = ROOT / "output" / "thirsty-wombat2" / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR thirsty-wombat2 推理")
    parser.add_argument("--image", type=Path, required=True, help="待检测图片")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练得到的权重")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="标注图保存路径")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"找不到权重: {args.checkpoint}，请先运行 scripts/train_thirsty_wombat.py")
    if not args.image.is_file():
        raise SystemExit(f"找不到图片: {args.image}")

    model = RFDETR.from_checkpoint(args.checkpoint)
    detections = model.predict(str(args.image), threshold=args.threshold)
    class_names = detections.data.get("class_name")
    if class_names is None:
        labels = [str(class_id) for class_id in detections.class_id]
    else:
        labels = [str(name) for name in class_names]

    image = detections.metadata["source_image"]
    annotated = sv.BoxAnnotator().annotate(image, detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

    output = args.output or (DEFAULT_OUTPUT / f"{args.image.stem}_pred.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(annotated)).save(output)
    print(f"检出 {len(labels)} 个目标: {labels}")
    print(f"标注图: {output}")


if __name__ == "__main__":
    main()
