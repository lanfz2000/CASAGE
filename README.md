# CASAGE

**Learning Where to Read and How Cells Differ for Single-Cell Gene Expression Prediction from Histology**

CASAGE predicts gene expression for cells at supplied coordinates in an H&E tile. The model adapts local feature sampling to each cell and uses relative expression preservation (REP) and global expression supervision (GES) as auxiliary objectives. Inference takes the image and cell coordinates.

## Setup

The reference run used Python 3.10, PyTorch 2.9.0, and two NVIDIA H200 GPUs. Install a CUDA-enabled PyTorch build and run these commands from the repository root:

```bash
python scripts/setup_cello.py
python -m pip install -r .cello/requirements.txt
```

The setup script fetches a pinned [CELLO](https://github.com/zjgao02/CELLO) checkout and applies the CASAGE implementation. The base data loader and metric definitions follow CELLO. The default backbone is [Virchow2](https://huggingface.co/paige-ai/Virchow2). Obtain access to the model and sign in to Hugging Face before training. To use a local timm-compatible copy of the encoder, set `CASAGE_ENCODER_DIR` to its directory.

## Train and evaluate

Use the provided [10X-Xenium-52 benchmark](https://huggingface.co/datasets/gaozijun/cello_data), or point `--data_root` to an existing copy. The full download takes about 40 GB. Run the data and training steps on a machine with enough storage and two CUDA GPUs.

```bash
python .cello/scripts/download_data.py --out_dir data/xenium52 --no_st
CUDA_VISIBLE_DEVICES=0,1 python .cello/scripts/train.py --config configs/casage_virchow2.yaml
python .cello/scripts/evaluate.py \
  --checkpoint outputs/casage_virchow2/latest_model.pt \
  --data_root data/xenium52 \
  --split_csv data/xenium52/splits/train_val_test_split.csv \
  --samples_csv data/xenium52/samples.csv \
  --rankings_dir data/xenium52/gene_rankings \
  --split test --batch_size 32 \
  --out_csv outputs/casage_virchow2/eval_test.csv
```

The configuration uses a global batch size of 256 and stops after five full training epochs. The evaluator writes per-slide results, ID/OOD means, and a separate per-gene CSV. The source repository contains no benchmark images, expression measurements, or trained checkpoints.

Code for adaptive aggregation, REP, and GES is under `casage/`. The patch in `patches/` connects these modules to CELLO's encoder, dataset, and benchmark evaluation interfaces.
