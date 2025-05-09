import argparse
from CPTSuperLearn.runner import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CPTSuperLearn model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (required)")
    args = parser.parse_args()
    train_model(args.config)
