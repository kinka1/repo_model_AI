"""
Script to sync/upsert all AI models from the MODEL_REGISTRY into the ai_models database table.
Reads metrics from models/metrics_summary.json and matches them by model name/architecture.

Usage:
    python scripts/sync_models_to_db.py

Environment:
    Reads DATABASE_URL from .env file (same as the main app).
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    print("❌ DATABASE_URL not found in .env file!")
    exit(1)

METRICS_FILE = ROOT_DIR / "models" / "metrics_summary.json"

# Model registry mirror (matching app/main.py MODEL_REGISTRY)
MODELS = [
    {
        "key": "scenario_1",
        "model_name": "Simple CNN (From Scratch)",
        "architecture": "SimpleCNN",
        "model_file_path": "models/best_model_cnn.pth",
    },
    {
        "key": "scenario_2",
        "model_name": "Simple CNN (With Augmentation)",
        "architecture": "SimpleCNN",
        "model_file_path": "models/best_model_cnn.pth",
    },
    {
        "key": "resnet50",
        "model_name": "ResNet50 (Transfer Learning + Fine-tuning)",
        "architecture": "ResNet50",
        "model_file_path": "models/best_model_resnet50.pth",
    },
    {
        "key": "resnet101",
        "model_name": "ResNet101 (Transfer Learning + Fine-tuning)",
        "architecture": "ResNet101",
        "model_file_path": "models/best_model_resnet101.pth",
    },
    {
        "key": "vgg16",
        "model_name": "VGG16 (Transfer Learning + Fine-tuning)",
        "architecture": "VGG16",
        "model_file_path": "models/best_model_vgg16.pth",
    },
    {
        "key": "vgg19",
        "model_name": "VGG19 (Transfer Learning + Fine-tuning)",
        "architecture": "VGG19",
        "model_file_path": "models/best_model_vgg19.pth",
    },
    {
        "key": "densenet121",
        "model_name": "DenseNet121 (Transfer Learning + Fine-tuning)",
        "architecture": "DenseNet121",
        "model_file_path": "models/best_model_densenet121.pth",
    },
    {
        "key": "efficientnet_b0",
        "model_name": "EfficientNet-B0 (Transfer Learning + Fine-tuning)",
        "architecture": "EfficientNet-B0",
        "model_file_path": "models/best_model_efficientnet_b0.pth",
    },
    {
        "key": "efficientnet_b3",
        "model_name": "EfficientNet-B3 (Transfer Learning + Fine-tuning)",
        "architecture": "EfficientNet-B3",
        "model_file_path": "models/best_model_efficientnet_b3.pth",
    },
]

# Production model key from .env
PRODUCTION_MODEL_KEY = os.getenv("PRODUCTION_MODEL_KEY", "efficientnet_b3")


def load_metrics() -> dict:
    """Load and parse metrics_summary.json, returning a dict keyed by architecture name."""
    if not METRICS_FILE.exists():
        print(f"⚠️  Metrics file not found: {METRICS_FILE}")
        return {}

    try:
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  Could not parse metrics file: {e}")
        return {}

    model_name_raw = data.get("model", "")
    test_metrics = data.get("test_metrics", {})

    accuracy = test_metrics.get("accuracy")
    f1 = test_metrics.get("f1")
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")

    # Map architecture name from metrics to our models
    # metrics_summary.json has "model": "ResNet50" -> matches architecture "ResNet50"
    print(f"📊 Loaded metrics for: {model_name_raw}")
    print(f"   Accuracy: {accuracy:.4%}" if accuracy else "   Accuracy: N/A")
    print(f"   F1 Score: {f1:.4%}" if f1 else "   F1 Score: N/A")

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    print("=" * 60)
    print("🔁 Syncing AI Models to Database")
    print("=" * 60)

    # Load shared metrics
    shared_metrics = load_metrics()
    print()

    # Connect to database
    engine = create_engine(DB_URL)
    print(f"✅ Connected to database")

    # Set the production model as is_active = True
    active_key = PRODUCTION_MODEL_KEY

    with engine.begin() as conn:
        # First, deactivate all models
        conn.execute(text("UPDATE ai_models SET is_active = false WHERE model_type = 'Gram Classification'"))
        print("🔓 All Gram Classification models deactivated")

        for item in MODELS:
            # Get metrics for this model (all models share the same metrics_summary.json)
            acc = shared_metrics.get("accuracy")
            f1 = shared_metrics.get("f1_score")
            prec = shared_metrics.get("precision")
            rec = shared_metrics.get("recall")

            is_active = (item["key"] == active_key)

            # Check if model already exists
            existing = conn.execute(
                text("""
                    SELECT id FROM ai_models
                    WHERE model_name = :name
                    ORDER BY id LIMIT 1
                """),
                {"name": item["model_name"]},
            ).fetchone()

            params = {
                "model_name": item["model_name"],
                "model_type": "Gram Classification",
                "version": "1.0",
                "model_file_path": item["model_file_path"],
                "accuracy": acc,
                "f1_score": f1,
                "precision_score": prec,
                "recall_score": rec,
                "is_active": is_active,
            }

            if existing:
                conn.execute(
                    text("""
                        UPDATE ai_models
                        SET model_type         = :model_type,
                            version            = :version,
                            model_file_path    = :model_file_path,
                            accuracy           = :accuracy,
                            f1_score           = :f1_score,
                            precision_score    = :precision_score,
                            recall_score       = :recall_score,
                            is_active          = :is_active,
                            updated_at         = NOW()
                        WHERE id = :id
                    """),
                    {**params, "id": existing[0]},
                )
                status = "✅" if is_active else "🔄"
                print(f"  {status} Updated: {item['model_name']}")
            else:
                conn.execute(
                    text("""
                        INSERT INTO ai_models
                            (model_name, model_type, version, model_file_path,
                             accuracy, f1_score, precision_score, recall_score,
                             is_active, created_at, updated_at)
                        VALUES
                            (:model_name, :model_type, :version, :model_file_path,
                             :accuracy, :f1_score, :precision_score, :recall_score,
                             :is_active, NOW(), NOW())
                    """),
                    params,
                )
                status = "✅" if is_active else "  "
                print(f"  {status} Inserted: {item['model_name']}")

    print()
    print(f"⭐ Active production model: {active_key}")
    print("=" * 60)
    print("Done! 🎉")


if __name__ == "__main__":
    main()
