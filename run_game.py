from game.cpt_game import CPTGame


if __name__ == "__main__":
    validation_data_folder = "data/vali"
    model_path = "results"

    game = CPTGame(validation_data_folder, model_path)
    game.run()