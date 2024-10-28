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
- Open `dataset/dataloader.py` and replace path_to_data with your local dataset path.
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
```

### 2. Training

Before running the training scripts, you need to update the dataset loading paths:

#### **Update Dataset Paths**

- Open `dataset/dataloader.py` and replace `path_to_data` with your local dataset path.

#### **Run Training Scripts**

Execute the following commands to train the models for both long-term and short-term predictions across the three datasets. The trained models will be saved in the `ckpt` directory with the specified model names.
```bash
python main_h36m.py --model_save_dir ckpt --model_save_name h36m_short_model --task short

python main_h36m.py --model_save_dir ckpt --model_save_name h36m_long_model --task long

python main_cmu.py --model_save_dir ckpt --model_save_name cmu_short_model --task short

python main_cmu.py --model_save_dir ckpt --model_save_name cmu_long_model --task long

python main_3dpw.py --model_save_dir ckpt --model_save_name 3dpw_short_model --task short

python main_3dpw.py --model_save_dir ckpt --model_save_name 3dpw_long_model --task long
```

## License
This project is licensed under the MIT License.

## Contact
If you have any questions or suggestions, feel free to reach out:

Email: 766368242@qq.com
GitHub: zlf766368242
