import argparse
from CPTSuperLearn.runner import validate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the CPTSuperLearn model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (required)")
    args = parser.parse_args()
    validate_model(args.config)