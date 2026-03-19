"""
Entry point for the Toronto Fire Response Time project.

Commands
--------
python main.py                         # run full data pipeline only
python main.py --train                 # pipeline → train model
python main.py --train --no-tune       # pipeline → train (skip hyperparameter search)
python main.py --explain               # generate explainability plots (model must exist)
python main.py --train --explain       # pipeline → train → explain
python main.py --predict data/new.csv  # predict on a processed CSV
"""

import argparse
from src.pipeline import run_pipeline, run_training, run_explainability


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toronto Fire Response Time — data pipeline & ML model"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the XGBoost model after running the data pipeline"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning (faster, uses default params)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate feature importance and SHAP explainability plots"
    )
    parser.add_argument(
        "--predict",
        metavar="CSV_PATH",
        default=None,
        help="Path to a processed CSV to run predictions on"
    )
    parser.add_argument(
        "--output",
        metavar="OUT_PATH",
        default=None,
        help="Output path for predictions CSV (used with --predict)"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the data pipeline step (useful if data is already processed)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Data pipeline
    if not args.skip_pipeline and args.predict is None:
        run_pipeline()

    # 2. Train
    if args.train:
        run_training(tune=not args.no_tune)

    # 3. Explain
    if args.explain:
        run_explainability()

    # 4. Predict
    if args.predict:
        from src.modeling.predict import predict_from_processed_csv
        predict_from_processed_csv(
            input_csv=args.predict,
            output_csv=args.output,
        )


if __name__ == "__main__":
    main()