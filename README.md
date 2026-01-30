# Learned HeyFusion

Welcome to the official GitHub repository for the open-source software and hardware associated with our article. This repository provides all the necessary code, designs, and documentation to reproduce the results and experiments described in our paper.

## About the Project

In this study, we introduce a novel shared-control system for key-hole docking operations, combining a commercial camera with occlusion-robust pose estimation and a hand-eye information fusion technique. This system is used to enhance docking precision and force-compliance safety. To train a hand-eye information fusion network model, we generated a self-supervised dataset using this docking system. After training, our pose estimation method showed improved accuracy compared to traditional methods, including observation-only approaches, hand-eye calibration, and conventional state estimation filters. In real-world phantom experiments, our approach demonstrated its effectiveness with reduced position dispersion (1.23 ± 0.81 mm vs. 2.47 ± 1.22 mm) and force dispersion (0.78 ± 0.57 N vs. 1.15 ± 0.97 N) compared to the control group. These advancements in semi-autonomy co-manipulation scenarios enhance interaction and stability. The study presents an anti-interference, steady, and precision solution with potential applications extending beyond laparoscopic surgery to other minimally invasive procedures.


## Paper Information

For a detailed explanation of the methods and results, please refer to our paper:

- **Title**: Semi-Autonomous Laparoscopic Robot Docking with Learned Hand-Eye Information Fusion
- **Authors**: Huanyu Tian, Martin Huber, Christopher E. Mower, Zhe Han, Changsheng Li, Xingguang Duan, and Christos Bergeles
- **Journal**: Accepted by T-BME
- **arXiv Link**: [https://arxiv.org/abs/2405.05817](https://arxiv.org/abs/2405.05817)

## Video Demonstration

A video demonstration of the project is available at the following link:

- **Video Link**: [https://youtu.be/M_uZHY2E7gY](https://youtu.be/M_uZHY2E7gY)

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- OpTas
- CasaDi
- OpenCV
- ROS2 Humble
- LBR_FRI_LIB





### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/learned-heyfusion.git
    cd learned-heyfusion
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

- **Training with KalmanNet**:  
  To train the network using the KalmanNet model, run the following command from the repository root:
    ```bash
    python train/trainModel.py
    ```

- **Training with KalmanNetOrigin**:  
  Alternatively, to train using the original variant (KalmanNetOrigin), run:
    ```bash
    python train/trainOrigin.py
    ```

### Project Structure

The repository is organized into the following folders:

- **config**: Configuration and parameter settings (e.g., `config/config.py`).
- **dataset**: Data processing and dataset utilities (e.g., `dataset/dataset_alb.py` and `dataset/finding_ground_truth.py`).  
  **Note:** The actual data files should be stored in a folder named `data` (at the same level as this README) with subfolders named from `0` to `18`.
- **models**: Model definitions and filtering implementations (e.g., `models/nnkf.py`, `models/SystemModel.py`, and `models/UKF.py`).
- **pipeline**: The training and evaluation pipeline (e.g., `pipeline/pipeline.py`).
- **train**: Training scripts (e.g., `train/trainModel.py` and `train/trainOrigin.py`).

## Data Notes (CSV Requirements)

Training scripts expect per-trial CSV files at:

```
data/0/saved_state.csv
data/1/saved_state.csv
...
data/18/saved_state.csv
```

Each CSV must be **headerless** and **numeric-only** (float values). One row corresponds to one timestamp. The intended column layout is:

```
0-2   observation position (x, y, z)
3-6   observation quaternion (qw, qx, qy, qz)
7-12  twist (vx, vy, vz, wx, wy, wz)
13-14 reprojection errors (two scalars)
15-17 ground-truth position (x, y, z)
18-21 ground-truth quaternion (qw, qx, qy, qz)
```

Minimum required columns: **22**. The training input is built as:
`[observation_pose(7), reprojection_error(1), twist(6)]` → total 14 values,
and the training target is the ground-truth pose (7 values).



## Contact

For any inquiries, please contact [huanyu.tian@kcl.ac.uk](mailto:huanyu.tian@kcl.ac.uk).
