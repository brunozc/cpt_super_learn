import os
import argparse
from game.cpt_game import CPTGame

def main():
    parser = argparse.ArgumentParser(description='CPT Digital Twin Game')
    parser.add_argument('--data', type=str, default='data/vali',
                        help='Path to validation data folder')
    parser.add_argument('--model', type=str, default='results_2',
                        help='Path to trained model folder')
    parser.add_argument('--width', type=int, default=1200,
                        help='Game window width')
    parser.add_argument('--height', type=int, default=800,
                        help='Game window height')

    args = parser.parse_args()

    # Check if data and model paths exist
    if not os.path.exists(args.data):
        print(f"Error: Data path '{args.data}' does not exist")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' does not exist")
        return

    # Launch the game
    game = CPTGame(args.data, args.model, args.width, args.height)
    game.run()

if __name__ == "__main__":
    main()
