# Phi-SegNet
The manuscript is currently under review, source code will be updated soon.


Clean PyTorch implementation of [**Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation**](https://arxiv.org/abs/2601.16064).

The repository is intentionally compact. The main components can be copied independently into another project:

```text
phisegnet/model.py          # model only
phisegnet/loss.py           # losses only
phisegnet/dataset.py        # dataset + paired transforms
phisegnet/evaluation.py     # metrics + test loop
phisegnet/prepare_dataset.py# raw data split/preparation
```

## Installation

```bash
pip install -r requirements.txt
```

## Expected dataset format

The training and testing scripts expect paired images and binary masks:

```text
data/DatasetName/
├── Train/
│   ├── Main/
│   └── Mask/
├── Val/
│   ├── Main/
│   └── Mask/
└── Test/
    ├── Main/
    └── Mask/
```

Images and masks should have matching filenames or matching filename stems.

## Prepare data

Random split:

```bash
python scripts/prepare_data.py \
  --raw-image-dir raw/images \
  --raw-mask-dir raw/masks \
  --output-dir data/BUSI \
  --split-ratio 0.8 0.1 0.1 \
  --seed 42
```

Fixed split using CSV:

```bash
python scripts/prepare_data.py \
  --raw-image-dir raw/images \
  --raw-mask-dir raw/masks \
  --output-dir data/BUSI \
  --split-csv splits/busi_split.csv
```

The CSV should contain:

```csv
filename,split
case001.png,train
case002.png,val
case003.png,test
```

## Train

Edit `configs/phisegnet.yaml`, then run:

```bash
python scripts/train.py --config configs/phisegnet.yaml
```

The best and last checkpoints are saved to the configured output folder.

## Test

```bash
python scripts/test.py \
  --checkpoint outputs/phisegnet_busi/best_model.pth \
  --image-dir data/BUSI/Test/Main \
  --mask-dir data/BUSI/Test/Mask \
  --save-dir outputs/test_results
```

The test script saves:

```text
outputs/test_results/
├── results.csv
├── summary.json
└── predictions/
```

## Use the model in another project

```python
from phisegnet.model import PhiSegNet

model = PhiSegNet(input_channels=3, num_classes=1, pretrained=True)
logits, decoder_layers, merged_features = model(images)
```

## Use the loss in another project

```python
from phisegnet.loss import PhiSegLoss

criterion = PhiSegLoss(phase_weight=0.01, jaccard_weight=1.0, fft_size=(256, 256))
loss, loss_dict = criterion(logits, decoder_layers, masks)
```

## Citation

Please cite the paper if this code helps your research.
@article{ali2026phisegnet,
  title={Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation},
  author={Ali, Shams Nafisa and Hasan, Taufiq},
  journal={arXiv preprint arXiv:2601.16064},
  year={2026},
  url={https://arxiv.org/abs/2601.16064}
}


