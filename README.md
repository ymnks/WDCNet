# WDCNet: Wavelet-aggregated Dual-stream Network for Long-term Time Series Forecasting

> **Status: 🚧 Under active development — skeleton release only.**
>
> This repository currently contains **only the directory skeleton and documentation**.
> The full source code, training scripts, configuration files and pretrained checkpoints
> will be released **after the paper is accepted**.
>
> We are actively maintaining this repository; the structure below reflects the final
> layout. Please ⭐ watch this repo if you would like to be notified when the full
> code drops.

---

## 1. About

WDCNet is a wavelet-based model for long-term multivariate time series forecasting.
The pipeline consists of four stages:

1. **Multi-level stationary wavelet transform (SWT)** decomposes the lookback window
   into one approximation plus several detail sub-series of equal length;
2. **Multi-level Wavelet Reconstruction (MWR)** aggregates cross-scale information
   between sub-series with paired-channel grouped convolutions;
3. **Dual-stream interaction** couples a historical-encoding branch with a learnable
   prediction branch through repeated cross-temporal enhancers;
4. **Inverse SWT** fuses the per-branch predictions back into the time domain.

The model is evaluated on seven public benchmarks: ETTh1, ETTh2, ETTm1, ETTm2,
Weather, Electricity and Traffic.

---

## 2. Repository status & roadmap

| Stage | Contents | Status |
|---|---|---|
| **Now** | Directory skeleton + this document | ✅ released |
| **Paper accepted** | Full model code, training/testing entry points, configs, reproduction scripts | ⏳ pending |
| **Later** | Pretrained checkpoints, per-run logs, figure-generation scripts | ⏳ pending |

> **Why a skeleton first?** The manuscript is currently under review. We prefer to
> publish a stable, documented code base **once** rather than a moving target.
> The directory layout below is final — files will be dropped into it as-is.

---

## 3. Directory structure

```
WDCNet/
├── model/              # model definitions (to be released)
├── data_provider/      # dataset loading & train/val/test splitting
├── experiments/        # training / validation / testing loops
├── utils/              # metrics, schedulers, early stopping, time features
├── scripts/            # experiment launchers & reproduction scripts
├── dataset/            # (empty) place datasets here — see §5
├── docs/               # supplementary documentation
├── README.md
├── requirements.txt
└── LICENSE
```

### 3.1 `model/`

| File | Description |
|---|---|
| `__init__.py` | Wavelet operator interface used by the model |
| `myModel.py` | **WDCNet** (main model): SWT → MWR → dual-stream interaction → iSWT |
| `myModeloM.py` | Ablation: w/o MWR |
| `myModeloC.py` | Ablation: w/o cross-temporal enhancer |
| `myModeloB.py` | Ablation: single-direction gated Mamba |
| `myModelSSV.py` | Ablation: single-stream variant |
| `myModelLIV.py`, `myModelDIV.py` | Ablation: interaction-placement variants |
| `myModelNoGate.py`, `myModelBiAdd.py` | Ablation: gating variants |
| `DLinear.py`, `PatchTST.py`, `iTransformer.py` | Baseline models used for comparison |

### 3.2 `data_provider/`

| File | Description |
|---|---|
| `data_factory.py` | Builds train/val/test `DataLoader`s from CLI arguments |
| `data_loader.py` | Dataset classes (ETT / custom CSV), sliding-window construction, scaling |

### 3.3 `experiments/`

| File | Description |
|---|---|
| `exp_basic.py` | Base experiment class: device placement, model/optimizer construction |
| `exp_long_term_forecasting.py` | Main training / validation / test loop (early stopping, LR schedule, metric aggregation) |
| `exp_long_term_forecasting_partial.py` | Partial-training entry (used by one supplementary experiment) |

### 3.4 `utils/`

| File | Description |
|---|---|
| `tools.py` | Early stopping, learning-rate schedulers, plotting helpers |
| `metrics.py` | MSE / MAE / RMSE / MAPE / MSPE |
| `timefeatures.py` | Calendar-feature encoding |
| `masking.py` | Attention masks |
| `context.py` | Small context helper |

### 3.5 `scripts/`

Experiment launchers and reproduction helpers (multi-GPU queues, array of configs).
A curated subset will be released; ad-hoc debugging scripts are excluded.

### 3.6 Root-level entry points

| File | Description |
|---|---|
| `run.py` | Main entry point (`--is_training 1` to train, `0` to test) |
| `probe_recipe_fast.py` | Lightweight single-seed probe for quick configuration screening |

---

## 4. Environment (planned)

The paper's experiments were conducted on a single NVIDIA GPU with PyTorch.
Key dependencies:

```
python >= 3.10
torch >= 2.0
numpy, pandas, scikit-learn
PyWavelets
einops
mamba-ssm, causal-conv1d
ptwt == 1.0.1        # install with: pip install ptwt==1.0.1 --ignore-requires-python
```

> ⚠️ `ptwt` **must be ≥ 1.0** — earlier versions do not export the stationary
> wavelet transform (`ptwt.swt` / `ptwt.iswt`) used by this model.
> Exact version pins will be shipped together with the full code.

---

## 5. Datasets

No data is stored in this repository. Please download the public benchmarks and
place them under `dataset/`:

```
dataset/
├── ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
├── weather/weather.csv
├── electricity/electricity.csv
└── traffic/traffic.csv
```

- ETT / Weather / Electricity / Traffic are publicly available
  (see the links in the original DLinear / iTransformer / TimeXer repositories).
- All datasets use the standard chronological split (6:2:2) and are standardized
  with statistics fitted on the training portion only.

---

## 6. Reproduction (after release)

Planned usage once the code is public:

```bash
# train
python run.py --is_training 1 --model myModel --data ETTh1 \
    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 192 \
    --w_level 2 --num_blocks 2 --s_ff 64 \
    --loss L1 --lradj cosine --clip_grad 1 \
    --train_epochs 20 --patience 3 --batch_size 16
```

A `scripts/` launcher reproducing every table/figure of the paper will be provided.

---

## 7. Citation

```bibtex
@inproceedings{wdcnet2026,
  title     = {WDCNet: Wavelet-aggregated Dual-stream Network for Long-term Time Series Forecasting},
  author    = {TBD},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

The BibTeX entry will be finalized upon acceptance.

---

## 8. Contact

For questions about the method or the upcoming release, please open an issue
in this repository.

---

## 中文说明

**当前状态：仓库仍处于持续开发与维护中。**

目前仅公开**目录骨架与文档**，完整的模型代码、训练脚本、配置文件和预训练权重
将在**论文正式录用之后**一次性完全公开。这样安排是为了让对外发布的版本保持一致、
可复现，而不是中途反复变动。

- 目录结构与最终版本完全一致，后续只需把对应文件放入即可；
- 论文所用的核心组件：多级平稳小波变换（SWT）分解 → 多级小波重构（MWR）
  跨尺度聚合 → 双流交互 → 逆小波（iSWT）融合；
- 依赖中 `ptwt` 版本需 ≥ 1.0（该版本才提供 `swt` / `iswt` 接口）；
- 数据集请自行下载至 `dataset/`，仓库不存放任何数据；
- 如需提前了解实现细节或开展合作，欢迎通过 issue 联系。

我们会持续更新本仓库（包括目录细化与文档补充），**论文过审后完全公开**。
