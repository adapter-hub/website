---
title: "Adapters Updates: ReFT, QLoRA, Merging, New Models & Hub"
date: 2024-07-22
authors:
  - name: Clifton Poth
    twitter: "@clifapt"
  - name: Leon Engländer
    twitter: "@LeonEnglaender"
  - name: Timo Imhof
    twitter: "@timo_imhof"
  - name: Hannah Sterz
    twitter: "@h_sterz"
  - name: Jonas Pfeiffer
    twitter: "@PfeiffJo"
summary: |
  Today we are releasing the newest updates in our Adapters library. This post summarizes new features in the latest release as well as selected new features since our initial release in Nov 2023, including new adapter methods, new supported models and Hub updates.
---

Eight months ago, [we released _Adapters_](https://adapterhub.ml/blog/2023/11/introducing-adapters/), our new unified library for parameter-efficient and modular fine-tuning.
_Adapters_ stands in direct tradition to our work on `adapter-transformers` since 2020, the first open-source library for parameter-efficient fine-tuning.
Since its initial release, _Adapters_ has received various updates, the newest being released today.
In this post, we'll go through some of the most exciting new features released today and in the last few months.
You can find the full list of changes in the latest release [in our release notes]().

[TOC]

You can find _Adapters_ [on GitHub](https://github.com/Adapter-Hub/adapters) or install it via pip:

```bash
pip install -U adapters
```

## Representation Fine-Tuning (ReFT)

<div align="center">
<figure text-align="center">
<img src="/static/images/reft.jpg" height="200">
  <figcaption text-align="center">
    Illustrations from the ReFT paper (Wu et al., 2024): Left: The general framework of applying ReFT interventions. Right: Visualization of LoReFT.
  </figcaption>
 </figure>
</div>

Representation Fine-Tuning (ReFT), proposed by [Wu et al. (2024)](https://arxiv.org/pdf/2404.03592), is a novel efficient adapter method.
It leverages so-called interventions to adapt the pre-trained representations of a language model.
Within the context of ReFT, these interventions can intuitively be thought of as adapter modules placed after each Transformer layer.
In the general form, an intervention function $\Phi$ can thus be defined as follows:

$$
\Phi(h) = h + R^T (W h + b - R h)
$$

Here, $R \in \mathbb{R}^{r \times d}$ and $W \in \mathbb{R}^{r \times d}$ are low-rank matrices of rank $r$.
$h$ is the layer output hidden state at a single sequence position, i.e. interventions can be applied independently at each position.

Based on this general form, the ReFT paper proposes multiple instantiations of ReFT methods supported by _Adapters_:

- **LoReFT** enforces orthogonality of rows in $R$. Defined via [`LoReftConfig`](adapters.LoReftConfig) or via the `orthogonality` attribute as in the following example:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=True
)  # equivalent to LoreftConfig()
```

- **NoReFT** does not enforce orthogonality in $R$. Defined via [`NoReftConfig`](adapters.NoReftConfig) or equivalently:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=False
)  # equivalent to NoReftConfig()
```

- **DiReFT** does not enforce orthogonality in $R$ and additionally removes subtraction of $R h$ in the intervention, Defined via [`DiReftConfig`](adapters.DiReftConfig) or equivalently:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=False, subtract_projection=False
)  # equivalent to DiReftConfig()
```

In addition, _Adapters_ supports configuring multiple hyperparameters tuned in the ReFT paper in `ReftConfig`, including:

- `prefix_positions`: number of prefix positions
- `suffix_positions`: number of suffix positions
- `layers`: The layers to intervene on. This can either be `"all"` or a list of layer ids
- `tied_weights`: whether to tie parameters between prefixes and suffixes

You can use ReFT adapters exactly as any other adapter type in _Adapters_:

```python
from adapters import AutoAdapterModel, LoReftConfig

model = AutoAdapterModel.from_pretrained("roberta-base")

config = LoReftConfig()
model.add_adapter("loreft_adapter", config=config)
model.train_adapter("loreft_adapter")
# add training loop ...
```

## Adapter Merging

## Quantized Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb)

Quantization of model weights has become an important method for drastically reducing the memory footprint of recent large language models.
Quantizing parameters down to 8 bits or 4 bits ([Dettmers & Zettlemoyer, 2023](https://arxiv.org/pdf/2212.09720)) have enabled running large models on limited hardware with minimal performance reduction.

While initially limited to model inference, **QLoRA** ([Dettmers et al., 2023](https://arxiv.org/pdf/2305.14314)) has proposed combining model quantization with adapter training using LoRA.

QLoRA combines several innovations to reduce the memory footprint while fine-tuning a large LM. In short:

- _4-bit NormalFloat quantization_ reduces the size of the base model to 4 bits per parameter while optimizing for maximizing the retained information.
- _Double quantization_ additionally quantizes constants required for quantization for additional memory saving.
- _Paged optimizers_ offloads optimizer states into CPU memory when they don't fit into GPU memory and automatically reloads them when needed.
- _LoRA training_ fine-tunes LoRA layers on the task while keeping the quantized base model weights fixes.

Make sure to check out [the paper](https://arxiv.org/pdf/2305.14314) for detailed explanations!
The figure below visualizes the key differences between full fine-tuning, LoRA and QLoRA:

<div align="center">
<figure text-align="center">
<img src="/static/images/qlora.jpg">
  <figcaption text-align="center">
    Illustration from the QLoRA paper (Dettmers et al., 2023) comparing full fine-tuning, LoRA and QLoRA.
  </figcaption>
 </figure>
</div>

Model quantization and paged optimizers are integrated to the Transformers library via the **[bitsandbytes library](https://github.com/TimDettmers/bitsandbytes)**:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    torch_dtype=torch.bfloat16,
)
```

Since _Adapters_ v0.2.0, all adapter implementations integrate seamlessly with quantized models, e.g. for QLoRA:

```python
import adapters
from adapters import LoRAConfig

adapters.init(model)

config = LoRAConfig(alpha=16, r=64)
model.add_adapter("qlora", config=config)
model.train_adapter("qlora")
```

Note you can easily swap out the adapter config here! You can not only train QLoRA, but also QBottleneck adapters, QPrefixTuning and more!

For a full guide, check out our [Notebook tutorial for quantized fine-tuning of Llama](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb).

## New Models

### Whisper

### Other Models

TODO
Models added since initial release:

- MT5
- Whisper
- Mistral
- PLBart

## Hub Updates

Within the last few weeks, we have archived the "original" Hub repository (found at: [Adapter-Hub/Hub](https://github.com/adapter-hub/Hub)) released alongside our initial AdapterHub release in 2020.
The Hub repository on GitHub is now in read-only mode, meaning no new adapters can be added there.

It's recommended to upload all new adapters to the Hugging Face Model Hub, which will be the only supported Hub for _Adapters_ in the future ([Learn more](https://docs.adapterhub.ml/huggingface_hub.html)). We have moved all ~300 publicly accessible adapters, including all of our original collection and most third-party contributions over to the Hugging Face Model Hub. Check out our Hugging Face Hub page at: [https://huggingface.co/AdapterHub](https://huggingface.co/AdapterHub).

In v1.0 of Adapters, attempting to load adapters from the original Hub repo will automatically redirect to loading the same adapter from the Hugging Face Model Hub.
There is no breaking change in loading an adapter ID, the same adapter weights will be loaded.
However, some parameters related to Hub loading and discovery have been deprecated or removed.
Learn more about breaking changes [here](https://github.com/adapter-hub/adapters/discussions/725).

## Citation

```bibtex
@inproceedings{poth-etal-2023-adapters,
    title = "Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning",
    author = {Poth, Clifton  and
      Sterz, Hannah  and
      Paul, Indraneil  and
      Purkayastha, Sukannya  and
      Engl{\"a}nder, Leon  and
      Imhof, Timo  and
      Vuli{\'c}, Ivan  and
      Ruder, Sebastian  and
      Gurevych, Iryna  and
      Pfeiffer, Jonas},
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.13",
    pages = "149--160",
}
```
