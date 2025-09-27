# <p align=center>`LVP for Domain-Adaptive Semantic Segmentation`</p><!-- omit in toc -->

## Table of Contents

  * [Introduction](#1-introduction)
  * [Environment Setup](#2-Environment-Setup)
  * [Dataset](#3-Dataset-Setup)
  * [Framework Structure](#4-Framework-Structure)
  * [Acknowledgements](#5-Acknowledgements)
  * [Future Work](#6-FutureWork)


## Code Implementation Statement

As discussed in [8](https://github.com/lhoyer/MIC/issues/8), [54](https://github.com/lhoyer/MIC/issues/54) and [63](https://github.com/lhoyer/MIC/issues/63), **our method inherits the instability of [MIC](https://github.com/lhoyer/MIC).** :cry:

To address this, we will release two versions of the code: one designed to produce more stable results with lower standard deviation, and another that achieves higher performance albeit with greater variance. :smiley:

Note, however, that the mathematical expectation of performance is the same for both, i.e., **76.9%** mIoU and **69.9%** mIoU on GTAV→Cityscapes and SYNTHIA→Cityscapes, respectively. :100:

## 1. Introduction

 🔥 Pending


## 2. Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/LVP-UDASeg
source ~/venv/LVP-UDASeg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

## 3. Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.


The final folder structure should look like this:

```none
LVP
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## 4. Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).


🔑 **Key Idea**

Our Language-Vision Prior (LVP) combines:

* Language Prior (LP): multi-prototype prompts capture class-level semantics and intra-class variance.

* Vision Prior (VP): bi-directional masking encourages robust global-local reasoning.

Together, they guide stable and reliable domain adaptation.

**Overall Training Pseudocode**

```python
# ===== Overall Training with Language & Vision Priors (LVP) =====
# Task: Unsupervised Domain Adaptive Semantic Segmentation
# Loss: L_total = L_ce + α L_da + (L_ta + β L_pa) + (L_pc + γ L_rc)   

# -----------------------------
# 0) Preparation
# -----------------------------
# Inputs:
#   DS = {(x_s, y_s)}          # labeled source data
#   DT = {x_t}                 # unlabeled target data
#   Classes = [c_1, ..., c_C]  # class names
# Hyper-params:
α = 1.0     # balance for UDA (self-training / adversarial) loss  
β = 0.01    # weight for prototype assignment loss                 
γ = 0.25    # weight for reconstruction consistency                
K = 5       # prototypes per class                                 
m = 16      # learnable context length for prompts                 
τ = 0.1     # temperature for cosine-similarity losses             

# Models:
#   gθ        : feature encoder
#   h_cls     : segmentation head
#   h_proj    : projection head for pixel embeddings
#   h_rec     : reconstruction head (for VP)
#   TextEncoder(·): frozen text encoder (e.g., CLIP)               

# Teacher (EMA):
#   f_φ = h_cls ∘ g_{θ̄}   # teacher used to produce target pseudo-labels  

# -----------------------------
# 1) Build Language Prototypes (LP)
# -----------------------------
# Learnable contexts: Z_k ∈ R^{m×D}, k = 1..K
Z = {Z_k for k in range(1, K+1)}  # learnable

# For each class c, build K prompt variants and encode to get textual prototypes
P = {}  # P[c] = [p_{c,1}, ..., p_{c,K}]
for c in Classes:
    P[c] = []
    for k in range(1, K+1):
        t_ck = concat(Z_k, embedding(c))          # tc,k = [Z_k, e_c]          
        p_ck = TextEncoder(t_ck)                   # pc,k = TextEncoder(tc,k)   
        P[c].append(p_ck)

# -----------------------------
# 2) Training Loop
# -----------------------------
for step in range(max_iters):

    # ---- Sample mini-batch ----
    (x_s, y_s) ~ DS
    x_t ~ DT

    # ---- Supervised on source ----
    feat_s = gθ(x_s)
    logits_s = h_cls(feat_s)
    L_ce = CrossEntropy(logits_s, y_s)            # supervised CE on source     

    # ---- Pseudo-labels on target (teacher EMA) ----
    with no_grad():
        logits_t_teacher = f_φ(x_t)
        y_hat_t, q_t = ArgmaxWithConfidence(logits_t_teacher)   # labels + confidence
    # Self-training style adaptation loss (e.g., CE weighted by confidence)      
    feat_t = gθ(x_t)
    logits_t = h_cls(feat_t)
    L_da = WeightedCE(logits_t, y_hat_t, weight=q_t)

    # ---- Language Prior losses (LP) ----
    # Pixel embeddings for LP alignment
    V_s = h_proj(feat_s)      # pixel-wise visual embeddings (source)
    V_t = h_proj(feat_t)      # pixel-wise visual embeddings (target)
    V_all, Y_all = concat(V_s, V_t), concat(y_s, y_hat_t)

    # (a) Online clustering within each class via optimal transport (Sinkhorn)    
    #     Assign each pixel embedding v to one of K prototypes of its class.
    assignments = {}
    for c in Classes:
        V_c = select_by_class(V_all, Y_all, c)
        # assignments[c]: one-hot over {1..K} for each pixel of class c
        assignments[c] = SinkhornCluster(V_c, P[c])                                   # 

    # (b) Textual Alignment loss (inter-class): pull v to closest prototype of its class,
    #     push away closest prototypes of other classes                               
    L_ta = TextualAlignmentLoss(V_all, Y_all, P, temperature=τ)

    # (c) Prototype Assignment loss (intra-class): pull v to its assigned prototype,
    #     push away other prototypes (same- & cross-class)                            
    L_pa = PrototypeAssignmentLoss(V_all, Y_all, P, assignments, temperature=τ)

    # ---- Vision Prior losses (VP) ----
    # Reliability map from pseudo-label confidence (encourage masking uncertain/rare)  
    R = ReliabilityMapFromConfidence(logits_t_teacher, neigh_radius=3, thresh=0.968)  # Eq.(11)

    # Build bi-directional progressive masks (in→out / out→in)                         
    regions = PartitionIntoRings(x_t, num_regions=4)                                   # Fig.4
    mask_in  = ProgressiveMask(regions, order="in_out",  reliability_map=R,
                               mask_ratios=[0.65, 0.70, 0.70, 0.75])                   # 
    mask_out = ProgressiveMask(regions, order="out_in", reliability_map=R,
                               mask_ratios=[0.75, 0.70, 0.70, 0.65])                   # dual

    # Randomly choose one painting (mutually exclusive)                                
    xP = ApplyMask(x_t, choice(mask_in, mask_out, p_out_in=0.4))                       # ε=0.4

    # Consistency to full-image prediction                                             
    logits_mask = h_cls(gθ(xP))
    L_pc = WeightedCE(logits_mask, y_hat_t, weight=q_t)                                # prediction consistency

    # Reconstruction consistency                                                       
    xR = h_rec(gθ(xP))
    L_rc = L1(xR, x_t)

    # ---- Total loss ----
    L_total = L_ce + α * L_da + (L_ta + β * L_pa) + (L_pc + γ * L_rc)

    # ---- Optimize student, update teacher with EMA ----
    Optimize(L_total, params=[gθ, h_cls, h_proj, h_rec, Z])
    UpdateEMA(teacher=f_φ, student=(gθ, h_cls))

# End for
```

##  5. Acknowledgements

TIP is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)


## Future Work

## Multi-Prototype Representation

Current results indicate that **small-object classes** (e.g., *traffic light*, *traffic sign*, *pole*) show higher intra-class diversity, while **large-area classes** (e.g., *road*, *sky*) appear more homogeneous. Using a single prototype per class may not be sufficient to capture such diversity.

### Directions

- **Adaptive Prototype Allocation**
  - Allocate prototypes per class based on:
    - *Intra-class diversity* (e.g., covariance trace, mean pairwise distance).
    - *Effective sample size* (e.g., log of pixel count).
    - *Resource budget* (global prototype limit with min/max constraints).

- **Dynamic Selection**
  - Explore automatic methods to determine prototype counts:
    - *k-means* with silhouette or Davies–Bouldin scores.
    - *Gaussian Mixture Models* with BIC/AIC.

- **Class-Specific Strategies**
  - Small-object classes with heterogeneous appearance → more prototypes.
  - Large-object classes with stable texture → fewer prototypes.

- **Evaluation Metrics**
  - Monitor **intra-class coverage** (distance to nearest prototype).
  - Monitor **inter-class separation** (margin to non-class prototypes).
  - Use these signals to refine prototype allocation.

---

*The goal is to better capture intra-class variability without overspending resources, paving the way for finer-grained representation and improved segmentation quality.*



# Prototype Allocation

This repository provides a utility function to allocate prototype counts per class  
based on intra-class diversity and sample size.

## Example: Allocate Prototypes

```python

import torch
import math

def allocate_prototypes(feats_by_class, K_total, K_min=1, K_max=10, alpha=0.7, beta=0.3, eps=1e-8):
    """
    Allocate prototype counts per class based on intra-class diversity and sample size.

    Args:
        feats_by_class (dict[int, torch.Tensor]): A dictionary mapping class -> features (N_c, C).
        K_total (int): Total number of prototypes across all classes.
        K_min (int): Minimum number of prototypes per class (default=1).
        K_max (int): Maximum number of prototypes per class (default=10).
        alpha (float): Weight for diversity in allocation (default=0.7).
        beta (float): Weight for sample count in allocation (default=0.3).
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        dict[int, int]: A dictionary mapping each class to its allocated number of prototypes.
    """
    classes = sorted(feats_by_class.keys())
    D, L = [], []  # store diversity and log-count values

    for c in classes:
        X = feats_by_class[c]
        # Use covariance trace as a measure of diversity
        Xc = X - X.mean(dim=0, keepdim=True)
        cov_trace = (Xc.T @ Xc / max(1, X.shape[0]-1)).diag().sum().item()
        D.append(max(cov_trace, 0.0))
        L.append(math.log1p(X.shape[0]))  # log(1 + sample size)

    # Normalize diversity and sample size contributions
    D_sum = sum(D) + eps
    L_sum = sum(L) + eps
    d_hat = [d / D_sum for d in D]
    n_hat = [l / L_sum for l in L]

    # Initial allocation: ensure each class has at least K_min
    base = K_min * len(classes)
    room = max(K_total - base, 0)
    q = [alpha * d + beta * n for d, n in zip(d_hat, n_hat)]
    q_sum = sum(q) + eps
    k_float = [K_min + room * (qi / q_sum) for qi in q]  # float allocation

    # Round allocations and apply min/max limits
    k_round = [int(round(x)) for x in k_float]
    k_round = [max(K_min, min(K_max, k)) for k in k_round]

    # Adjust to make sure the total sum equals K_total
    diff = K_total - sum(k_round)
    if diff != 0:
        # Priority: adjust classes whose rounded value deviates most from float target
        prio = sorted(
            range(len(classes)),
            key=lambda i: (k_float[i] - k_round[i]),
            reverse=(diff > 0),
        )
        i = 0
        while diff != 0 and i < len(prio):
            idx = prio[i]
            newk = k_round[idx] + (1 if diff > 0 else -1)
            if K_min <= newk <= K_max:
                k_round[idx] = newk
                diff += -1 if diff > 0 else 1
            i += 1

    K_dict = {c: k for c, k in zip(classes, k_round)}
    return K_dict

```




## Code Availability Statement
This code is associated with a paper currently under review. To comply with the review process, the code will be made FULLY available once the paper is accepted. 

We appreciate your understanding and patience. Once the code is released, we will warmly welcome any feedback and suggestions. Please stay tuned for our updates!

