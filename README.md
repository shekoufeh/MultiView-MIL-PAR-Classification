# Foundation-Model-Guided Multi-View MIL for Automated Orthodontic Assessment

This repository contains the PyTorch implementation for the paper:  
**"Foundation-Model-Guided Multi-View Multiple Instance Learning for Automated Orthodontic Assessment"** 

The framework introduces a robust pipeline for automated orthodontic diagnostic assessment by leveraging large-scale Vision Foundation Models (DINOv2/v3) and a Multi-View Multiple Instance Learning (MIL) architecture to handle disparate clinical perspectives (Frontal, Lateral, and Occlusal).

---

##  Pipeline Overview

The pipeline is designed to mirror the clinical workflow of an orthodontist, processing multiple intra-oral views to reach a global diagnosis.

1.  **Intelligent-Cropping:** Foundation-model-guided tooth localization using DINO-based feature clustering and cluster selection.
2.  **Feature Extraction:** Utilizing DINOv2 backbone for robust, high-dimensional clinical representations.
3.  **Multi-View Fusion:** Aggregating features from disparate anatomical views using Multiple Instance Learning (MIL) or alternative fusion strategies (Input/Early/Late).

---

##  Running the Code

### 1. Environment Setup
We recommend using Conda to manage dependencies and ensure reproducibility across different hardware setups.
```bash
# Create the environment from the provided yaml files
conda env create -f environment.yml

# Activate the environment
conda activate dinov3_ortho
```

### 2. Preprocessing: Intelligent-Cropping and Montage Generation
You can standardize your raw dental images into a single 2x3 layout for Input Fusion strategy using the scripts in "utils":
```bash
python multiview_grid_gen.py --input ./../data/all --output ./../data/montages
```
To apply adaptive cropping to focus on the dental region:
```bash
python intelligent_cropping.py  --repo_dir ./path/to/dinov3_repo  --weights ./weights/dinov3_vitb16.pth --input_dir ./../data/all --output_dir ./../data/smart_crop
```
Ensure your filenames follow the convention {PatientID}_{View}.png e.g. <XXXX>_right.png, <XXXX>_left.png, <XXXX>_center.png, <XXXX>_up.png, and <XXXX>_down.png.

### 3. Training
The repository provides the primary MIL framework as well as several baseline fusion strategies discussed in the paper.

Train the primary Multi-view MIL model using:
```bash
python MV_MIL_Classification.py --architecture vit_small_patch14_dinov2.lvd142m --loss_fn LabelSmoothing --lr 0.001 --drop 0.05 --num_epochs 50 --batch_size 32  --usecrop True --fold <fold_num> 
```

To train a model with Early/Late or Input Fusion strategies use:
```bash
# for early or late fusion 
python early_late_fusion.py --fusion <early/late>  --architecture vit_small_patch14_dinov2.lvd142m --loss_fn LabelSmoothing --lr 0.001 --drop 0.05 --num_epochs 50 --batch_size 32 --usecrop True --fold <fold_num>

# for Input fusion
python input_fusion.py --architecture vit_small_patch14_dinov2.lvd142m --lr 0.0001 --drop 0.05 --num_epochs 50 --batch_size 16 --fold <fold_num>
```

### 4. Evaluation
To evaluate the models as well as code for computing inter-rater agreement are provided in "utils" and can be used as follows:
```bash
python final_evaluation.py
python inter_rater_reliability.py
```
## Data Format and Privacy
Due to patient privacy, the full dataset used in this study cannot be publicly released. To facilitate reproduction of the pipeline, we provide a sample directory structure and file formats used for this project. In the annotation csv files, classes 0 and 1 correspond to PAR <= 10 and class 2 corresponds to PAR > 10. See "data" folder.

## Publication and Citation
The paper related to this work is currently under peer review. If you use our software as part of a scientific project, please cite the corresponding publication:
```bibtex
@article{anonymous2026foundation,
  title={Foundation-Model-Guided Multi-View Multiple Instance Learning for Automated Orthodontic Assessment},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```
## Author
[Shekoufeh Gorgi Zadeh](https://github.com/shekoufeh)
