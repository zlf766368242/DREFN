# DREFN

## Description

This project implements long-term and short-term prediction models across three datasets: H36M, CMU, and 3DPW. Utilizing CUDA 12.1 and PyTorch 2.3.1, the models are trained and evaluated to assess their performance in various prediction tasks. The repository includes scripts for data loading, training, and testing, ensuring a streamlined workflow for researchers and developers.

## Dependencies

- **CUDA 12.1**: Ensure that CUDA 12.1 is installed on your system to leverage GPU acceleration.
- **Python 3.12**: The project is developed using Python version 3.12.
- **PyTorch 2.3.1**: Deep learning framework used for model development and training.

## Usage

### 1. Evaluation

Before running the evaluation scripts, update the dataset loading paths and pre-trained model paths:

#### **Update Paths**
- Open `dataset/dataloader.py` and replace `path_to_data` with your local dataset path.
- Open `test.py` and replace `weights_path_j` with your local pre-trained model path.

#### **Run Evaluation Scripts**
Execute the following commands to validate the model's performance on long-term and short-term predictions across three datasets:

```bash
python test_h36m.py --task short
python test_h36m.py --task long
python test_cmu.py --task short
python test_cmu.py --task long
python test_3dpw.py --task short
python test_3dpw.py --task long
