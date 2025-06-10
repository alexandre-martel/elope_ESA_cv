# EV Flownet Lightning

## Overview
The EV Flownet Lightning project is designed for processing event-based data to predict velocities using a neural network architecture. The project utilizes PyTorch and PyTorch Lightning for efficient model training and validation.

## Project Structure
```
EV_flownet_lightning/
├── src/
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── model.py             # Contains the neural network model definitions
│   ├── dataset.py           # Handles loading and processing of event data
│   ├── train.py             # Sets up the training process
│   └── utils.py             # Contains utility functions for data processing
├── requirements.txt          # Lists project dependencies
└── README.md                 # Documentation for the project
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage
1. Prepare your event-based data in the required format.
2. Modify the `train.py` file if necessary to adjust parameters such as batch size or learning rate.
3. Run the training script:

```
python src/train.py
```

## Dependencies
This project requires the following Python packages:
- PyTorch
- PyTorch Lightning
- NumPy
- TensorBoard

Make sure to install these packages before running the code.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.