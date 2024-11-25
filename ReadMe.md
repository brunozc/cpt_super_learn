# CPTSuperLearn

## Objective

CPTSuperLearn is a Deep Reinforcement Learning framework designed to improve subsurface in-situ testing.
The objective of this project is to train and evaluate a Deep Q-Learning (DQL) agent to optimize the placement
of in-situ testing (e.g. Cone Penetration Test) points in a given environment.
The agent learns to minimize the Root Mean Square Error (RMSE) between the true and predicted data
while considering the cost of sampling.

![DRL](DRL.png)


## Installation

### Prerequisites

- Python 3.12 or higher
- The data files for training and validation are available on [Zenodo](https://zenodo.org/records/13143431/files/data.zip).
- If you want to use the data from the schemaGan, you need to download the [schemaGan model](https://zenodo.org/records/13143431/files/schemaGAN.h5).
- To automatically download the data files and the schemaGan model, run the following script:
    ```sh
    python download_data.py
    ```

### Steps to use CPTSuperLearn

1. Clone the repository:
    ```sh
    git clone https://github.com/brunozc/CPTSuperLearn.git
    cd CPTSuperLearn
    ```

1. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

1. Install the required dependencies:
    ```sh
    pip install .
    ```

    or alternatively if you want to edit or change the code:
    ```sh
    pip install -r requirements.txt
    ```


## Usage

### Training the Model

To train the DQL agent, run the [`train_model`](train_model.py) script:

```sh
python train_model.py
```

This will train the model using the data in the [`./data/train`](./data/train) folder and save the results in the results folder.

### Evaluating the Model

To evaluate the trained model, run the [`validate_model`](validate_model.py) script:

```sh
python validate_model.py
```

This will evaluate the model using the data in the [`data/val`](data/val/) folder and save the validation scores and RMSE values in the results/validation folder.


### Plotting Results
To visualize the training scores, run the [`plot_score`](plot_score.py) script:

```sh
python plot_score.py
```

To visualize the validation results, run the [`plot_validation`](plot_validation.py) script:

```sh
python plot_validation.py
```

### Project Structure

```plaintext
CPTSuperLearn
├── CPTSuperLearn
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   ├── interpolator.py
│   └── utils.py
├── train_model.py
├── validate_model.py
├── plot_score.py
├── plot_validation.py
```

* CPTSuperLearn: Contains the core modules for the environment, agent, interpolators, and utilities.
* data: Contains the data files for training and validation.
* results: Contains the output results from training and validation.
* train_model.py: Script to train the DQL agent.
* validate_model.py: Script to evaluate the trained model.
* plot_score.py: Script to plot the training scores.
* plot_validation.py: Script to plot the validation results.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

