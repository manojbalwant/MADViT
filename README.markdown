# MADViT: Multilayer Distillation for Anomaly Detection in Aerial Imagery

This repository implements **MADViT**, a Vision Transformer-based framework for unsupervised anomaly detection in aerial imagery, as presented in the paper:

> **MADViT: A Vision Transformer-Based Multilayer Distillation Framework for Explainable Anomaly Detection in Aerial Imagery**\
> Manoj Kumar Balwant, Shivendu Mishra, Rajiv Misra\
> Link to paper *(actual link to be provide)*

## Introduction

MADViT uses a teacher-student distillation approach with Vision Transformers (ViTs) to detect anomalies in aerial imagery without requiring labeled data. It offers explainable results through anomaly maps and achieves competitive performance on datasets like Drone-Anomaly and UIT-ADrone.

### Key Features

- Unsupervised anomaly detection using normal images only.
- Multi-layer distillation for efficient knowledge transfer.
- Patch-based anomaly maps for interpretability.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MADViT.git
   cd MADViT
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train the model on the Drone-Anomaly dataset:

```bash
python train_val_VIT_Distillation.py --data_path dataset/Drone-Anomaly/Bike Roundabout/ --save_path experiments/bike_roundabout/ --valid_file b_s1_te_01 --lr 5e-4 --wd 5e-2
```

### Evaluation

Evaluate a trained model:

```bash
python train_val_VIT_Distillation.py --train 0 --data_path dataset/Drone-Anomaly/Bike Roundabout/ --save_path experiments/bike_roundabout/
```

For a full tutorial, check out `notebooks/demo.ipynb`.

## Datasets

Supported datasets:

- Drone-Anomaly *(https://uit-together.github.io/datasets/UIT-ADrone/)*
- UIT-ADrone *(https://gitlab.lrz.de/ai4eo/reasoning/Drone-Anomaly)*

See `datasets/drone_anomaly.py` for data loading details.

<table style="min-width: 25px">
<colgroup><col style="min-width: 25px"></colgroup><tbody><tr class="border-border"><td colspan="1" rowspan="1"><h3 dir="ltr">Results</h3><p dir="ltr">Our method delivers strong performance on the following benchmark datasets:</p><h4 dir="ltr">Drone-Anomaly Dataset</h4><table style="min-width: 75px"><colgroup><col style="min-width: 25px"><col style="min-width: 25px"><col style="min-width: 25px"></colgroup><tbody><tr class="border-border"><th colspan="1" rowspan="1"><p dir="ltr">Scene</p></th><th colspan="1" rowspan="1"><p dir="ltr">AUC (%)</p></th><th colspan="1" rowspan="1"><p dir="ltr">EER</p></th></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">Highway</p></td><td colspan="1" rowspan="1"><p>88.33</p></td><td colspan="1" rowspan="1"><p>0.210</p></td></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">Bike Roundabout</p></td><td colspan="1" rowspan="1"><p>84.20</p></td><td colspan="1" rowspan="1"><p>0.250</p></td></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">Farmland Inspection</p></td><td colspan="1" rowspan="1"><p>79.65</p></td><td colspan="1" rowspan="1"><p>0.267</p></td></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">Solar Panel Inspection</p></td><td colspan="1" rowspan="1"><p>80.48</p></td><td colspan="1" rowspan="1"><p>0.276</p></td></tr></tbody></table><h4 dir="ltr">UIT-ADrone Dataset</h4><table style="min-width: 50px"><colgroup><col style="min-width: 25px"><col style="min-width: 25px"></colgroup><tbody><tr class="border-border"><th colspan="1" rowspan="1"><p dir="ltr">Metric</p></th><th colspan="1" rowspan="1"><p dir="ltr">Value</p></th></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">AUC (%)</p></td><td colspan="1" rowspan="1"><p>83.65</p></td></tr><tr class="border-border"><td colspan="1" rowspan="1"><p dir="ltr">EER</p></td><td colspan="1" rowspan="1"><p>0.2399</p></td></tr></tbody></table><p dir="ltr">These results highlight our method’s effectiveness in anomaly detection across diverse aerial scenarios, as reported in the research article.</p><p></p></td></tr></tbody>
</table>

### Anomaly Map Example

*(generate and upload this)*

## Citation

If you use this code, please cite:

```bibtex
@article{balwant2025madvit,
  title={MADViT: A Vision Transformer-Based Multilayer Distillation Framework for Explainable Anomaly Detection in Aerial Imagery},
  author={Balwant, Manoj Kumar and Mishra, Shivendu and Misra, Rajiv},
  journal={XXXX.XXXXX},
  year={2025}
}
```

## Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License. See LICENSE for details.