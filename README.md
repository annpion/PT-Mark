# ptMark

This repository package is organized around the main entry script:

`main/run_ptmark.py`

## Project Structure

```text
github/
├── main/
│   ├── run_ptmark.py
│   ├── ptmark_pipeline.py
│   ├── ptmark_reference_pipeline.py
│   ├── ptmark_watermark.py
│   ├── ptmark_null_inversion.py
│   ├── ptmark_optim_utils.py
│   ├── ptmark_prompt_utils.py
│   ├── ptmark_scheduling_ddim.py
│   └── ptmark_utils.py
├── loss/
│   ├── __init__.py
│   ├── pytorch_ssim.py
│   └── watson_vgg.py
├── example/config/
│   └── config.yaml
├── environment.yml
├── png_filenames.log
└── README.md
```

## 1. Create Environment

Use the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate ptmark
```

## 2. Prepare Required Files

Before running the code, prepare these resources locally:

- Stable Diffusion model weights
- image dataset for watermarking
- prompt file for DiffusionDB evaluation
- COCO metadata file if you run the COCO branch

The script expects paths like these inside the repository:

```text
github/
├── data/
│   ├── watermark_image/
│   ├── prompts/
│   │   └── eval.parquet
│   ├── coco/
│   │   └── meta_data.json
│   └── no_watermark_imgs/
│       └── v2.1-base/
└── outputs/
```

## 3. Edit Configuration

Update [config.yaml](/data/code/wyp/ZoDiac-master/github/example/config/config.yaml) before running:

- `model_id`: path to your Stable Diffusion model
- `model_xl_id`: path to your SDXL model if needed
- `save_img`: output directory
- watermark parameters such as `w_channel`, `w_radius`, and `w_seed`

Example:

```yaml
model_id: 'path/to/stable-diffusion-2-1-base'
model_xl_id: 'path/to/stable-diffusion-xl-base-1.0'
save_img: './outputs/ptmark'
```

## 4. Run the Code

Run the main script from the repository root:

```bash
python main/run_ptmark.py \
  --original_image_path ./data/watermark_image \
  --num_inner_steps 10 \
  --image_number 100 \
  --watarmark_loss_weight 0.0006
```

Arguments:

- `--original_image_path`: input image folder
- `--num_inner_steps`: number of inner optimization steps
- `--image_number`: number of images to process
- `--watarmark_loss_weight`: watermark loss weight

## 5. Output

The script writes generated results under:

```text
./outputs/null_optimization_double_channel_1/
```

It also reads auxiliary files from:

- `./png_filenames.log`
- `./data/prompts/eval.parquet`
- `./data/coco/meta_data.json`
- `./data/no_watermark_imgs/v2.1-base/`

## 6. Notes

- The current code assumes a CUDA-enabled PyTorch environment.
- Some external Python packages are required by the original environment file, including `diffusers`, `datasets`, `lpips`, and `open_clip`.
- If you only want to release code, do not upload model weights, datasets, logs, or generated images.
