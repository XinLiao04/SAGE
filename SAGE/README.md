# 🚀 SAGE: Synergistic Adaptive Gating of Experts

> **SAGE** is a multi-modal hateful video detection framework that dynamically fuses heterogeneous modality experts via **synergistic adaptive gating**, enabling robust and interpretable decision-making across text, audio, and visual signals.

<p align="center">
  <img src="https://img.shields.io/badge/Task-Hateful%20Video%20Detection-blueviolet" />
  <img src="https://img.shields.io/badge/Modality-Text%20%7C%20Audio%20%7C%20Video-orange" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Python-3.11-green" />
</p>

---

## 📁 Project Structure

```text
- config                  # experiment configuration
- model                   # model implementation
  - CustomDataset.py      # custom dataset module
  - CustomLoss.py         # loss function
  - Runner.py             # model runner
  - Sage.py               # SAGE model implementation
  - Trainer.py            # model trainer
  - utils.py              # training utils
- preprocess              # data preprocessing
  - audio-extract.py      # extract audios
  - frame-extract.py      # extract video frames
  - trans-extract.py      # extract transcripts
- model/Main.py           # entry point
```

---

## 📊 Dataset Available

Our experiments utilize two public multimodal hateful video datasets: **HateMM** and **MultiHateClip**.

### HateMM
In our experiment, we randomly split it into training / validation / test sets with a ratio of 7:1:2. 
Access the complete dataset from: [HateMM: A Multi-Modal Dataset for Hate Video Classification](https://github.com/hate-alert/HateMM)

### MultiHateClip
In our experiment, we use the official split strategy from the original work for fair comparison. 
Access the complete dataset from: [MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili](https://github.com/social-ai-studio/multihateclip)

---

## 🧠 Method Overview
<p align="center"> 
  <img src="assets/method_overview.png" width="85%" /> 
</p>

SAGE (Synergistic Adaptive Gating of Experts) adaptively integrates heterogeneous modality experts by learning instance-aware gating weights conditioned on cross-modal interactions.

Given text, audio, and visual representations extracted from pretrained encoders, SAGE:

- Projects modality-specific features into a shared latent space  
- Models cross-modal interactions among modality experts  
- Learns adaptive gating weights for each input instance  
- Produces a synergistic fused representation for final classification  

This design enables SAGE to emphasize informative modalities while suppressing noisy or misleading signals.

---

## ⚙️ Usage Guide

### 🧩 Requirements
To set up the environment dependencies (Python 3.11):

```bash
pip install -r requirements.txt
```

### 🔄 Dataset Preprocessing
Extract audio, 16 video frames, and corresponding transcripts.  
Modify the lines marked with `# To Do` according to your runtime environment.

```bash
python preprocess/audio-extract.py    # batch extract audios from videos
python preprocess/frame-extract.py    # batch extract video frames
python preprocess/trans-extract.py    # batch extract transcripts from audios
```

### 🛠️ Modify config.yaml Accordingly
Configure `config/config.yaml` according to your experiment settings before running.  
Ensure all paths to preprocessed data are correctly specified.

### ▶️ Start Training
Modify the lines marked with `# To Do` to run the pipeline:

```python
# load configuration file
with open("./config/config.yaml", "r") as f:        # To Do
    config = yaml.safe_load(f)

# run on HateMM
dataset = "hatemm"                                  # To Do
python Main.py

# run on MultiHateClip (Bilibili)
dataset = "mhclip-bl"                               # To Do
python Main.py

# run on MultiHateClip (YouTube)
dataset = "mhclip-yt"                               # To Do
python Main.py
```

During training, the model with the highest Macro-F1 score on the validation set is automatically saved for inference.