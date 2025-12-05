# 🌟 A reinforcement learning and pretrained network enhanced meta-learning framework for automated algorithm selection

A complete, fully reproducible pipeline for **meta-feature extraction**, **image feature generation**, **reinforcement learning–based feature selection**, and **final classifier training**.  
This repository includes both **preprocessed data** (`merge.pt`) and all source code for end‑to‑end experimentation.

---

## 🚀 Quick Start

We provide a fully processed merged dataset:

```
merge.pt
```

You can directly run the classifier:

```bash
python meta_classifier.py
```

---

# 🔧 Full Workflow (Train Everything from Scratch)

Follow the steps below if you want to regenerate all features and data.

---

## 1. 📁 Dataset Preparation

Dataset list:

```
dataset_list.txt
```

Download datasets from:

- UCI Machine Learning Repository  
- OpenML  

Performance results of algorithms on these datasets can be found at:

- http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/results.txt  
- OpenML repository  

Store all downloaded data inside:

```
dataset/
```

---

## 2. 🧮 Meta-Feature Extraction

Scripts are located in:

```
meta-features/
```

Files:

- `get_meta-features.py` — extract meta-features  
- `clean_missValue.py` — handle missing values  

Run:

```bash
python meta-features/get_meta-features.py
python meta-features/clean_missValue.py
```

---

## 3. 🖼️ Image Feature Extraction  
Includes t-SNE visualization & 256‑dimensional feature generation.

Scripts:

- `t-SNE.py`
- `256_features.py`

Run:

```bash
python pic/t-SNE.py
python pic/256_features.py
```

---

## 4. 🤖 DQN-Based Feature Selection

DQN-related scripts are located in:

```
DQN/
```

Workflow:

```bash
python DQN/reward/reward_train.py
python DQN/reward/reward_plot.py
python DQN/dqn_train.py
```

This will:

- Compute feature rewards  
- Visualize reward curves  
- Train DQN to generate the optimal feature subset  

---

## 5. 🔗 Merge All Features & Labels

Merge meta-features, image features, and labels into one file:

```bash
python merge.py
```

Output:

```
merge.pt
```

---

## 6. 🧪 Train & Evaluate the Classifier

Run:

```bash
python meta_classifier.py
```

---

# 📂 Project Structure

```
project/
│── dataset/
│── meta-features/
│   ├── get_meta-features.py
│   └── clean_missValue.py
│── pic/
│   ├── t-SNE.py
│   └── 256_features.py
│── DQN/
│   │── reward/
│       ├── reward_train.py
│       ├── reward_plot.py
│   └── dqn_train.py
│── merge.py
│── meta_classifier.py
│── merge.pt
│── dataset_list.txt
└── README.md
```


✨ Feel free to open issues or pull requests if you want to improve this work!
