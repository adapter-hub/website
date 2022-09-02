---
title: Updates in Adapter-Transformers v3.1
date: 2022-09-05
authors:
  - name: Clifton Poth
    twitter: "@clifapt"
summary: With the newest release of our `adapter-transformers` library, version 3.1, we take a further step towards integrating the diverse possibilities of parameter-efficient fine-tuning methods by supporting multiple new adapter methods and Transformer architectures.
---

![](/static/images/v3_methods.png "Illustration of efficient fine-tuning methods supported in v3 of adapter-transformers.")

Throughout the last few months, the field of parameter-efficient methods for fine-tuning Transformer-based models has seen a wide range of new innovations, proposing new adapter methods (e.g. [He et al., 2021](https://arxiv.org/pdf/2110.04366.pdf); [Liu et al., 2022](https://doi.org/10.48550/arXiv.2205.05638)) and applying them to new domains and tasks (e.g. [Chen et al., 2022](https://arxiv.org/pdf/2205.13535.pdf)).
With the newest release of our `adapter-transformers` library, version 3.1, we take a further step towards integrating the diverse possibilities of parameter-efficient fine-tuning methods by supporting multiple new adapter methods and Transformer architectures.

In the following sections, we highlight important new features and methods introduced with the new release.
The full changelog can be found [here](https://github.com/adapter-hub/adapter-transformers/releases/tag/adapters3.1.0).

[TOC]

You can find `adapter-transformers` [on GitHub](https://github.com/Adapter-Hub/adapter-transformers) or install it via pip:

```bash
pip install -U adapter-transformers
```

## New Adapter Methods

With [the release of `adapter-transformers` v3](https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/) a few months back, we started the process of integrating new adapter methods.
The new release v3.1 adds three new works that were released throughout the last year, namely _LoRA_ ([Hu et al., 2021](https://arxiv.org/pdf/2106.09685.pdf)), _UniPELT_ ([Mao et al., 2022](https://aclanthology.org/2022.acl-long.433.pdf)) and _(IA)^3_ ([Liu et al., 2022](https://doi.org/10.48550/arXiv.2205.05638)).

Previously, we have already integrated bottleneck adapters ([Houlsby et al., 2019](https://arxiv.org/pdf/1902.00751.pdf)), Prefix Tuning ([Li and Liang, 2021](https://aclanthology.org/2021.acl-long.353.pdf)), parallel adapters, Mix-and-Match adapters ([He et al., 2021](https://arxiv.org/pdf/2110.04366.pdf)) and Compacters ([Mahabadi et al., 2021](https://arxiv.org/pdf/2106.04647.pdf)).
For more on these methods, please refer [the blog post for the release of v3](https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/).
For a more general introduction to working with adapters, please refer to [our documentation](https://docs.adapterhub.ml/quickstart.html).

The following table compares the performance of our implementation of LoRA, (IA)^3 and bottleneck adapters, which are described in more detail afterwards, on the GLUE benchmark.
We use `roberta-base` as the base Transformer model and train for 20 epochs with learning rates of 1e-3, 1e-4 and 1e-4 for (IA)^3, LoRA and bottleneck adapters, respectively.

| Task      | Metric               |   (IA)^3 |   LoRA | Adapter (Houlsby)
|-----------|----------------------|-------|------------| --- |
| COLA      | Matthews Correlation | 52.14 |      58.35 | 59.81
| MNLI      | Accuracy             | 84.18 |      87.15 | 86.68
| MRPC      | F1                   | 87.9  |      90.63 | 90.53
| QNLI      | Accuracy             | 90.63 |      92.82 | 92.7
| QQP       | F1                   | 83.94 |      86.57 | 88.41
| RTE       | Accuracy             | 68.95 |      72.08 | 77.9
| SST2      | Accuracy             | 94.15 |      94.11 | 94.5
| STSB      | Spearmanr            | 88.03 |      89.82 | 90.58

### LoRA

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique proposed by [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf).
LoRA injects trainable low-rank decomposition matrices into the layers of a pre-trained model.
For any model layer expressed as a matrix multiplication of the form $h = W_0 x$, it therefore performs a reparameterization, such that:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

Here, $A \in \mathbb{R}^{r\times k}$ and $B \in \mathbb{R}^{d\times r}$ are the decomposition matrices and $r$, the low-dimensional rank of the decomposition, is the most important hyperparameter.

While, in principle, this reparameterization can be applied to any weights matrix in a model, the original paper only adapts the attention weights of the Transformer self-attention sub-layer with LoRA.
`adapter-transformers` additionally allows injecting LoRA into the dense feed-forward layers in the intermediate and output components of a Transformer block.
You can configure the locations where LoRA weights should be injected using the attributes in the [`LoRAConfig`](transformers.LoRAConfig) class.

_Example_:
```python
from transformers.adapters import LoRAConfig

config = LoRAConfig(r=8, alpha=16)
model.add_adapter("lora_adapter", config=config)
```

In the design of LoRA, Hu et al. (2021) also pay special attention to keeping the inference latency overhead compared to full fine-tuning at a minimum.
To accomplish this, the LoRA reparameterization can be merged with the original pre-trained weights of a model for inference.
Thus, the adapted weights are directly used in every forward pass without passing activations through an additional module.
In `adapter-transformers`, this can be realized using the built-in `merge_adapter()` method:
```python
model.merge_adapter("lora_adapter")
```

To continue training on this LoRA adapter or to deactivate it entirely, the merged weights first have to be reset again:
```python
model.reset_adapter("lora_adapter")
```

### (IA)^3

_Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3)_ is an efficient fine-tuning method proposed within the _T-Few_ fine-tuning approach by [Liu et al. (2022)](https://arxiv.org/pdf/2205.05638.pdf).
(IA)^3 introduces trainable vectors $l_W$ into different components of a Transformer model which perform element-wise rescaling of inner model activations.
For any model layer expressed as a matrix multiplication of the form $h = W x$, it therefore performs an element-wise multiplication with $l_W$, such that:

$$
h = l_W \odot W x
$$

Here, $\odot$ denotes element-wise multiplication where the entries of $l_W$ are broadcasted to the shape of $W$.

_Example_:
```python
from transformers.adapters import IA3Config

config = IA3Config()
model.add_adapter("ia3_adapter", config=config)
```

The implementation of (IA)^3, as well as the `IA3Config` class, are derived from the implementation of [LoRA](#lora), with a few main modifications.
First, (IA)^3 uses multiplicative composition of weights instead of additive composition as in LoRA.
Second, the added weights are not further decomposed into low-rank matrices.
Both of these modifications are controlled via the `composition_mode` configuration attribute by setting `composition_mode="scale"`.
Additionally, as the added weights are already of rank 1, `r=1` is set.

Beyond that, both methods share the same configuration attributes that allow you to specify in which Transformer components rescaling vectors will be injected.
Following the original implementation, `IA3Config` adds rescaling vectors to the self-attention weights (`selfattn_lora=True`) and the final feed-forward layer (`output_lora=True`).
Further, you can modify which matrices of the attention mechanism to rescale by leveraging the `attn_matrices` attribute.
By default, (IA)^3 injects weights into the key ('k') and value ('v') matrices, but not in the query ('q') matrix.

Finally, similar to LoRA, (IA)^3 also allows merging the injected parameters with the original weight matrices of the Transformer model.
E.g.:
```python
# Merge (IA)^3 adapter
model.merge_adapter("ia3_adapter")

# Reset merged weights
model.reset_adapter("ia3_adapter")
```

### UniPELT

An approach similar to the work of [He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) is taken by [Mao et al. (2022)](https://arxiv.org/pdf/2110.07577.pdf) in their _UniPELT_ framework.
They, too, combine multiple efficient fine-tuning methods, namely LoRA, Prefix Tuning and bottleneck adapters, in a single unified setup.
_UniPELT_ additionally introduces a gating mechanism that controls the activation of the different submodules.

Concretely, for each adapted module $m$, UniPELT adds a trainable gating value $\mathcal{G}_m \in (0, 1)$ that is computed via a feed-forward network ($W_{\mathcal{G}_m}$) and sigmoid activation ($\sigma$) from the Transformer layer input states ($x$):

$$\mathcal{G}_m \leftarrow \sigma(W_{\mathcal{G}_m} \cdot x)$$

These gating values are then used to scale the output activations of the injected adapter modules, e.g. for a LoRA layer:

$$
h \leftarrow W_0 x + \mathcal{G}_{LoRA} B A x
$$

In the configuration classes of `adapter-transformers`, these gating mechanisms can be activated via `use_gating=True`.
The full UniPELT setup can be instantiated using `UniPELTConfig`[^unipelt]:

[^unipelt]: Note that the implementation of UniPELT in `adapter-transformers` follows the implementation in the original code, which is slighlty different from the description in the paper. See [here](https://github.com/morningmoni/UniPELT/issues/1) for more.

```python
from transformers.adapters import UniPELTConfig

config = UniPELTConfig()
model.add_adapter("unipelt", config=config)
```

which is identical to the following `ConfigUnion`:

```python
from transformers.adapters import ConfigUnion, LoRAConfig, PrefixTuningConfig, PfeifferConfig

config = ConfigUnion(
    LoRAConfig(r=8, use_gating=True),
    PrefixTuningConfig(prefix_length=10, use_gating=True),
    PfeifferConfig(reduction_factor=16, use_gating=True),
)
model.add_adapter("unipelt", config=config)
```

Finally, as the gating values for each adapter module might provide interesting insights for analysis, `adapter-transformers` comes with an integrated mechanism of returning all gating values computed during a model forward pass via the `output_adapter_gating_scores` parameter:

```python
outputs = model(**inputs, output_adapter_gating_scores=True)
gating_scores = outputs.adapter_gating_scores
```
Note that this parameter is only available to base model classes and [AdapterModel classes](prediction_heads.md#adaptermodel-classes).
In the example, `gating_scores` holds a dictionary of the following form:
```text
{
    '<adapter_name>': {
        <layer_id>: {
            '<module_location>': np.array([...]),
            ...
        },
        ...
    },
    ...
}
```

## Further Updates

### New model integrations

Version 3.1 adds adapter support to the DeBERTa and Vision Transformer (ViT) architectures already integrated into HuggingFace Transformers.

The ViT integration is of particular interest as it opens the application area of our adapter implementations to the computer vision domains.
While most of the current work on adapter methods for Transformers happened in the NLP domain, adapters for Transformers in the vision domain have also been investigated recently ([He et al., 2022](https://arxiv.org/pdf/2203.16329.pdf); [Chen et al., 2022](https://arxiv.org/pdf/2205.13535.pdf)).

Below, we show some initial results of our ViT integration, using `google/vit-base-patch16-224` as the pre-trained base model:

Task | Full FT | Houlsby | Pfeiffer
--- | --- | --- | ---
CIFAR-10 | 98.88 | 98.72 | TBD
CIFAR-100 | 92.08 | 92.4 | TBD

All scores are accuracies on the evaluation set [^1].

### `adapter_summary()` method

The new release adds an `adapter_summary()` method that provides information on all adapters currently loaded into a base model in tabular form.
The method can be used as follows:

```python
model = AutoAdapterModel.from_pretrained("roberta-base")
for name, config in ADAPTER_CONFIG_MAP.items():
    model.add_adapter(name, config=config)
print(model.adapter_summary())
```

... which produces this output:

```text
================================================================================
Name                     Architecture         #Param      %Param  Active   Train
--------------------------------------------------------------------------------
pfeiffer                 bottleneck          894,528       0.718       0       1
houlsby                  bottleneck        1,789,056       1.435       0       1
pfeiffer+inv             bottleneck        1,190,592       0.955       0       1
houlsby+inv              bottleneck        2,085,120       1.673       0       1
compacter++              bottleneck           28,576       0.023       0       1
compacter                bottleneck           57,088       0.046       0       1
prefix_tuning            prefix_tuning     9,872,384       7.920       0       1
prefix_tuning_flat       prefix_tuning       552,960       0.444       0       1
parallel                 bottleneck        7,091,712       5.689       0       1
scaled_parallel          bottleneck        7,091,724       5.690       0       1
lora                     lora                294,912       0.237       0       1
ia3                      lora                 18,432       0.015       0       1
mam                      union            22,493,984      18.046       0       1
--------------------------------------------------------------------------------
Full model                               124,645,632     100.000               1
================================================================================
```

### Transformers upgrade

Version 3.1 of `adapter-transformers` upgrades the underlying HuggingFace Transformers library from v4.17.0 to v4.21.2, bringing many new features and bug fixes created by HuggingFace.

## References

- Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ArXiv, abs/2106.09685.
- Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. ArXiv, abs/2205.05638.
- Mao, Y., Mathias, L., Hou, R., Almahairi, A., Ma, H., Han, J., Yih, W., & Khabsa, M. (2021). UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning. ArXiv, abs/2110.07577.
- He, X., Li, C., Zhang, P., Yang, J., & Wang, X. (2022). Parameter-efficient Fine-tuning for Vision Transformers. ArXiv, abs/2203.16329.
- Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., & Luo, P. (2022). AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition. ArXiv, abs/2205.13535.

[^1]: Reported results for `adapter-transformers` only contain a single run each without hyperparameter tuning.
