"""在 thirsty-wombat2 上微调 RF-DETR Nano。

数据集已由 Ultralytics 从 ndjson 转成 YOLO 目录，RF-DETR 可直接读取：
D:\\ai_projects\\test\\rustinfer\\datasets\\thirsty-wombat2-c58c1a5e

用法（test-gpu 环境）:
    python scripts/train_thirsty_wombat.py
"""

from pathlib import Path

from rfdetr import RFDETRNano

DATASET_DIR = Path(r"D:\ai_projects\test\rustinfer\datasets\thirsty-wombat2-c58c1a5e")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "thirsty-wombat2"
RESUME = OUTPUT_DIR / "last.ckpt"


def main() -> None:
    model = RFDETRNano()
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=100,
        batch_size=4,
        grad_accum_steps=2,
        num_workers=2,
        lr=1e-4,
        output_dir=str(OUTPUT_DIR),
        resume=str(RESUME),
    )
    print(f"训练完成，权重目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
