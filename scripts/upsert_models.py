import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

ROOT = Path(r"D:\kerja\dataset")
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")

MODELS = [
    {"model_name": "ResNet50", "version": "1.0", "file": "models/best_model_resnet50.pth", "metrics": "models/metrics_resnet50.json"},
    {"model_name": "ResNet101", "version": "1.0", "file": "models/best_model_resnet101.pth", "metrics": "models/metrics_resnet101.json"},
    {"model_name": "EfficientNet-B0", "version": "1.0", "file": "models/best_model_efficientnet_b0.pth", "metrics": "models/metrics_efficientnet_b0.json"},
    {"model_name": "EfficientNet-B3", "version": "1.0", "file": "models/best_model_efficientnet_b3.pth", "metrics": None},
    {"model_name": "VGG16", "version": "1.0", "file": None, "metrics": None},
    {"model_name": "VGG19", "version": "1.0", "file": None, "metrics": None},
    {"model_name": "DenseNet121", "version": "1.0", "file": "models/best_model_densenet121.pth", "metrics": "models/metrics_densenet121.json"},
]


def load_metrics(rel_path):
    if not rel_path:
        return None, None
    path = ROOT / rel_path
    if not path.exists():
        return None, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    test_metrics = data.get("test_metrics")
    if isinstance(test_metrics, dict):
        acc = test_metrics.get("accuracy")
        f1 = test_metrics.get("f1")
        if f1 is None:
            f1 = test_metrics.get("f1_score")
        return acc, f1

    best_epoch = data.get("best_epoch")
    history = data.get("history") or []
    acc = None
    f1 = data.get("best_macro_f1")

    if best_epoch is not None:
        for row in history:
            if row.get("epoch") == best_epoch:
                acc = row.get("accuracy", acc)
                f1 = row.get("macro_f1", f1)
                break

    if acc is None and history:
        acc_vals = [r.get("accuracy") for r in history if r.get("accuracy") is not None]
        if acc_vals:
            acc = max(acc_vals)

    return acc, f1


def main():
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        for item in MODELS:
            accuracy, f1_score = load_metrics(item["metrics"])

            existing = conn.execute(
                text(
                    """
                    SELECT id
                    FROM ai_models
                    WHERE lower(model_name) = lower(:name) AND version = :version
                    ORDER BY id
                    LIMIT 1
                    """
                ),
                {"name": item["model_name"], "version": item["version"]},
            ).fetchone()

            params = {
                "model_name": item["model_name"],
                "model_type": "Gram Classification",
                "version": item["version"],
                "model_file_path": item["file"],
                "accuracy": accuracy,
                "f1_score": f1_score,
            }

            if existing:
                conn.execute(
                    text(
                        """
                        UPDATE ai_models
                        SET model_type = :model_type,
                            model_file_path = :model_file_path,
                            accuracy = :accuracy,
                            f1_score = :f1_score,
                            updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {**params, "id": existing[0]},
                )
                print(f"updated id={existing[0]} {item['model_name']}")
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO ai_models
                        (model_name, model_type, version, model_file_path, accuracy, f1_score, is_active, created_at, updated_at)
                        VALUES
                        (:model_name, :model_type, :version, :model_file_path, :accuracy, :f1_score, false, NOW(), NOW())
                        """
                    ),
                    params,
                )
                print(f"inserted {item['model_name']}")


if __name__ == "__main__":
    main()
