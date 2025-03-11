<div align="center">
<h1>Exploring Temporally-Aware Features for Point Tracking</h1>

[**Inès Hyeonsu Kim**](https://ines-hyeonsu-kim.github.io)<sup>1*</sup> · [**Seokju Cho**](https://seokju-cho.github.io)<sup>1*</sup> · [**Jiahui Huang**](https://gabriel-huang.github.io)<sup>2</sup> · [**Jung Yi**](https://github.com/YJ-142150)<sup>1</sup> · [**Joon-Young Lee**](https://joonyoung-cv.github.io)<sup>2</sup>  · [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>1</sup>

<sup>1</sup>KAIST AI&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Adobe Research

<span style="font-size: 1.5em;"><b>CVPR 2025</b></span>

<a href="https://arxiv.org/abs/2501.12218"><img src='https://img.shields.io/badge/arXiv-Chrono-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/Chrono/'><img src='https://img.shields.io/badge/Project_Page-Chrono-green' alt='Project Page'></a>


<!-- <p float='center'><img src="assets/teaser.png" width="80%" /></p> -->


</div>


Point tracking models often rely on **feature backbones that lack temporal awareness**, requiring computationally expensive refiners to correct errors and ensure coherence across frames. **What if your backbone itself could model long-term temporal dynamics?**

✨ **Introducing Chrono** – a **novel feature backbone** designed for point tracking, integrating a **long-range temporal adapter** for enhanced temporal consistency and efficiency.

## 🔍 Why Chrono?
🔨 **Filling the Gap:** Chrono addresses the **lack of temporally-aware feature backbones** and reduces reliance on **expensive refinement processes**.

⏳ **Long-Range Temporal Awareness:** Our **temporal adapter** enables feature extraction with **extended temporal context**, improving tracking quality.

⚡ **Smooth & Efficient Tracking:** Chrono produces **smoother initial tracks** in a **simple and effective** manner, **reducing the need for refiners**.

📈 **Refiner-Free Performance:** Chrono achieves **accuracy comparable to refiner-based pipelines**, proving that **temporally-aware features can be just as effective**.



## Environment

Prepare the environment by cloning the repository and installing the required dependencies:

```bash
git clone https://github.com/google-research/kubric.git

conda create -y -n chrono python=3.11
conda activate chrono

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip3 install -U lightning tensorflow_datasets tensorflow matplotlib mediapy tensorflow_graphics einops wandb
```

## Evaluation

#### 0. Evaluation Dataset Preparation
First, download the evaluation datasets:
```bash
# TAP-Vid-DAVIS dataset
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip

# TAP-Vid-RGB-Stacking dataset
wget https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip
unzip tapvid_rgb_stacking.zip
```
For downloading TAP-Vid-Kinetics, please refer to official [TAP-Vid repository](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).


#### 1. Download Pre-trained Weights

To evaluate Chrono on the benchmarks, first download the pre-trained weights.

| Model       | Pre-trained Weights |
|-------------|---------------------|
| Chrono (ViT-S) | [Link](https://drive.google.com/file/d/1Q-rqNl1ZkYhH4UtOjwcMH0oKkcCxMi7K/view?usp=sharing) |
| Chrono (ViT-B) | [Link](https://drive.google.com/file/d/1XYOr5pVncEAgyWcQZ_TjgvqLTcexdUQr/view?usp=sharing)  |

You can download the weights using the following commands:

```bash
pip install gdown

gdown 1Q-rqNl1ZkYhH4UtOjwcMH0oKkcCxMi7K
gdown 1XYOr5pVncEAgyWcQZ_TjgvqLTcexdUQr
```

#### 2. Adjust the Config File

In `config/dino.ini` (or any other config file), add the path to the evaluation datasets to `[TRAINING]-val_dataset_path`. Additionally, adjust the model size for evaluation in `[MODEL]-model_kwargs-model_size`.

#### 3. Run Evaluation

To evaluate the Chrono, use the `experiment.py` script with the following command-line arguments:

```bash
python experiment.py --config config/dino.ini --mode eval_{dataset_to_eval_1}_..._{dataset_to_eval_N}[_q_first] --ckpt_path /path/to/checkpoint --save_path ./path_to_save_checkpoints/
```

- `--config`: Specifies the path to the configuration file. Default is `config/dino.ini`.
- `--mode`: Specifies the mode to run the script. Use `eval` to perform evaluation. You can also include additional options for query first mode (`q_first`), and the name of the evaluation datasets. For example:
  - Evaluation of the DAVIS dataset: `eval_davis`
  - Evaluation of DAVIS and Kinetics in query first mode: `eval_davis_kinetics_q_first`
- `--ckpt_path`: Specifies the path to the checkpoint file. If not provided, the script will use the default checkpoint.
- `--save_path`: Specifies the path to save logs. 

Replace `/path/to/checkpoint` with the actual path to your checkpoint file. This command will run the evaluation process and save the results in the specified `save_path`.

## Training

#### Training Dataset Preparation

Download the panning-MOVi-E dataset used for training (approximately 273GB) from Huggingface using the following script. Git LFS should be installed to download the dataset. To install Git LFS, please refer to this [link](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux). Additionally, downloading instructions for the Huggingface dataset are available at this [link](https://huggingface.co/docs/hub/en/datasets-downloading).

```bash
git clone git@hf.co:datasets/hamacojr/LocoTrack-panning-MOVi-E
```

#### Training Script

Add the path to the downloaded panning-MOVi-E to the `[TRAINING]-kubric_dir` entry in `config/dino.ini` (or any other config file). Then, run the training with the following script:

```bash
python experiment.py --config config/dino.ini --mode train_davis --save_path ./path_to_save_checkpoints/
```

## 📚 Citing this Work
Please use the following bibtex to cite our work:
```
@article{kim2025exploring,
  title={Exploring Temporally-Aware Features for Point Tracking},
  author={Kim, In{\`e}s Hyeonsu and Cho, Seokju and Huang, Jiahui and Yi, Jung and Lee, Joon-Young and Kim, Seungryong},
  journal={arXiv preprint arXiv:2501.12218},
  year={2025}
}
```

## 🙏 Acknowledgement
This project is largely based on the [TAP repository](https://github.com/google-deepmind/tapnet) and [LocoTrack repository](https://github.com/cvlab-kaist/locotrack). Thanks to the authors for their invaluable work and contributions.
