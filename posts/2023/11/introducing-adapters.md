---
title: Introducing Adapters
date: 2023-11-22
authors:
  - name: Hannah Sterz
    twitter: "@h_sterz"
  - name: Clifton Poth, 
    twitter: "@clifapt"
  - name: Leon Engländer
    twitter: "@LeonEnglaender"
  - name: Timo Imhof
  - name: Jonas Pfeiffer
    twitter: "@PfeiffJo"
summary: |
  Introducing the new Adapters library the new package that supports adding parameter-efficient fine-tuning methods on top of transformers models and composition to achieve modular setups.
paper:
    citation: "Poth, C., Sterz, H., Paul, I., Purkayastha, S., Engländer, L., Imhof, T., Vuli´c, I., Ruder, S., Gurevych, I., & Pfeiffer, J. (2023). Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning."
    url: "https://arxiv.org/pdf/2311.11077.pdf"
---

We are happy to announce *Adapters*, the new library at the heart of the AdapterHub framework.
_Adapters_ stands in direct tradition to our previous work with the `adapter-transformers` library, while simultaneously revamping the implementation from ground up and smoothing many rough edges of the previous library.
This blog post summarizes the most important aspects of _Adapters_, as described in detail [in our paper](https://arxiv.org/abs/2311.11077) (to be presented as system demo at EMNLP 2023).

In summer of 2020, when we released the first version of _AdapterHub_, along with the `adapter-transformers` library, adapters and parameter-efficient fine-tuning[^peft] were still a niche research topic.
Adapters were first introduced to Transformer models in 2019 (Houlsby et al., 2019) and _AdapterHub_ was the very first framewerk to provide comprehensive tools for working with adapters, dramatically lowering the barrier of training own adapters or leveraging pre-trained ones.

[^peft]: We use the terms _parameter-efficient fine-tuning (PEFT)_ and _adapter_ interchangeably throughout this post and in all of our documents.

In the now more than three years following, _AdapterHub_ has increasingly gained traction within the NLP community, being [liked by thousands](https://github.com/adapter-hub/adapters/stargazers) and [used by hundreds](https://www.semanticscholar.org/paper/AdapterHub%3A-A-Framework-for-Adapting-Transformers-Pfeiffer-R%C3%BCckl%C3%A9/063f8b1ecf2394ca776ac61869734de9c1953808?utm_source=direct_link) for their resarch.
However, the field of parameter-efficient fine-tuning has grown even faster.
Nowadays, with recent LLMs growing ever larger in size, adapter methods, which do not fine-tune the full model, but instead only update a small number of parameters, have become increasingly main-stream.
[Multiple libraries, dozens of architectures and scores of applications](https://github.com/calpt/awesome-adapter-resources) compose a flourishing subfield of LLM research.

Besides parameter-efficiency, modularity is a second important characteristic of adapters [(Pfeiffer et al., 2023)](https://arxiv.org/pdf/2302.11529.pdf).
Sadly, this is overlooked by many existing tools.
From beginning on, _AdapterHub_ payed special attention to adapter modularity and composition, integrating setups like MAD-X (Pfeiffer et al., 2020).
_Adapters_ continues and expands this focus on modularity.

## The Library

_Adapters_ is a self-contained library supporting a diverse set adapter methods, integrating them into many common Transformer architectures and allowing flexible and complex adapter configuration.
Modular transfer learning can be achieved by combining adapters via six different composition blocks.

All in all, _Adapters_ offers substantial improvements compared to the predecessing `adapter-transformers` library:

1. Decoupled from the HuggingFace `transformers` library
2. Support of 10 adapter methods
3. Support of 6 composition blocks
4. Support of 20 diverse models

_Adapters_ can be easily installed via pip:
```bash
pip install adapters
```

The source code of _Adapters_ can be found [on GitHub](https://github.com/adapter-hub/adapters).

In the following, we highlight important components of _Adapters_.
If you have used `adapter-transformers` before, much of this will look familiar.

## Transformers Integration

_Adapters_ acts as an add-on to HuggingFace's Transformers library.
As a result, existing Transformers models can be easily attached with adapter functionality as follows:

```python
import adapters
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("t5-base")
adapters.init(model) # Adding adapter-specific functionality

model.add_adapter("adapter0")
```

However, we recommend using the model classes provided by *Adapters*, such as `XXXAdapterModel`, where "XXX" denotes the model architecture, e.g., Bert.
These models provide the adapter functionality without further initialization and support multiple heads, which is relevant when using composition blocks which can handle multiple outputs, for instance, the BatchSplit composition block. Here's an example of how to use such an `XXXAdapterModel` class:


```python
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("roberta-base")
model.add_adapter("adapter1", config="seq_bn") # add the new adapter to the model
model.add_classification_head("adapter1", num_classes=3) # add a sequence classification head

model.train_adapter("adapter1") # freeze the model weights and activate the adapter
```

## Adapter Methods

Each adapter method is defined by a configuration object or string, allowing for flexible customization of various adapter module properties, including placement, capacity, residual connections, initialization, etc. We distinguish between single methods consisting of one type of adapter module and complex methods consisting of multiple different adapter module types.

### Single Methods

*Adapters* supports single adapter methods that introduce parameters in new feed-forward modules such as bottleneck adapters (Houlsby et al., 2019), introduce prompts at different locations such as prefix tuning (Li and Liang, 2021), reparameterize existing modules such as LoRA (Hu et al., 2022) or re-scale their output representations such as (IA)³ (Liu et al., 2022). For more information see our [documentation](https://docs.adapterhub.ml/methods.html).

All adapter methods can be added to a model by the unified `add_adapter()` method, e.g.:

```python
model.add_adapter("adapter2", config="seq_bn")
```

Alternatively, a config class, along with custom parameters:

```python
from adapters import PrefixTuningConfig

model.add_adapter("adapter3", config=PrefixTuningConfig(prefix_length=20))
```

The following table gives an overview of all currently supported single methods, along with their configuration class and configuration string:

| Identifier | Configuration class | More information
| --- | --- | --- |
| `seq_bn` | `SeqBnConfig()` | [Bottleneck Adapters](https://docs.adapterhub.ml/methods.html#bottleneck-adapters) |
| `double_seq_bn` | `DoubleSeqBnConfig()` | [Bottleneck Adapters](https://docs.adapterhub.ml/methods.html#bottleneck-adapters) |
| `par_bn` | `ParBnConfig()` | [Bottleneck Adapters](https://docs.adapterhub.ml/methods.html#bottleneck-adapters) |
| `scaled_par_bn` | `ParBnConfig(scaling="learned")` | [Bottleneck Adapters](https://docs.adapterhub.ml/methods.html#bottleneck-adapters) |
| `seq_bn_inv` | `SeqBnInvConfig()` | [Invertible Adapters](https://docs.adapterhub.ml/methods.html#language-adapters---invertible-adapters) |
| `double_seq_bn_inv` | `DoubleSeqBnInvConfig()` | [Invertible Adapters](https://docs.adapterhub.ml/methods.html#language-adapters---invertible-adapters) |
| `compacter` | `CompacterConfig()` | [Compacter](https://docs.adapterhub.ml/methods.html#compacter) |
| `compacter++` | `CompacterPlusPlusConfig()` | [Compacter](https://docs.adapterhub.ml/methods.html#compacter) |
| `prefix_tuning` | `PrefixTuningConfig()` | [Prefix Tuning](https://docs.adapterhub.ml/methods.html#prefix-tuning) |
| `prefix_tuning_flat` | `PrefixTuningConfig(flat=True)` | [Prefix Tuning](https://docs.adapterhub.ml/methods.html#prefix-tuning) |
| `lora` | `LoRAConfig()` | [LoRA](https://docs.adapterhub.ml/methods.html#lora) |
| `ia3` | `IA3Config()` | [IA³](https://docs.adapterhub.ml/methods.html#ia-3) |
| `mam` | `MAMConfig()` | [Mix-and-Match Adapters](method_combinations.html#mix-and-match-adapters) |
| `unipelt` | `UniPELTConfig()` | [UniPELT](method_combinations.html#unipelt) |
| `prompt_tuning` | `LoRAConfig()` | [Prompt Tuning](https://docs.adapterhub.ml/methods.html#prompt_tuning) |

For more details on all adapter methods, visit [our documentation](https://docs.adapterhub.ml/methods.html).

### Complex Methods

While different efficient fine-tuning methods and configurations have often been proposed as standalone, combining them for joint training has proven to be beneficial (He et al., 2022; Mao et al., 2022). To make this process easier, Adapters provides the possibility to group multiple configuration instances using the `ConfigUnion` class. This flexible mechanism allows easy integration of multiple complex methods proposed in the literature (as the two examples outlined below), as well as the construction of other, new complex configurations currently not available nor benchmarked in the literature (Zhou et al., 2023).

**Mix-and-Match Adapters** (He et al., 2022) were proposed as a combination of Prefix-Tuning and parallel bottleneck adapters. Using `ConfigUnion`, this method can be defined as:

```python
from adapters import ConfigUnion, PrefixTuningConfig, ParBnConfig, AutoAdapterModel

model = AutoAdapterModel.from_pretrained("microsoft/deberta-v3-base")

adapter_config = ConfigUnion(
    PrefixTuningConfig(prefix_length=20),
    ParBnConfig(reduction_factor=4),
)
model.add_adapter("my_adapter", config=adapter_config, set_active=True)
```

**UniPELT** (Mao et al., 2022) combines LoRA, Prefix Tuning, and bottleneck adapters in a single unified setup. It additionally introduces a gating mechanism that controls the activation of the different adapter modules.

Learn more about complex adapter configurations using `ConfigUnion` [in our documentation](https://docs.adapterhub.ml/method_combinations.html).

## Modularity and Composition Blocks

![](/static/images/composition.png "Composition Blocks")

While the modularity and composability aspect of adapters have seen increasing interest in research, existing open-source libraries (Mangrulkar et al., 2022; Hu et al., 2023a) have largely overlooked these aspects. Adapters makes adapter compositions a central and accessible part of working with adapters by enabling the definition of complex, composed adapter setups. We define a set of simple composition blocks that each capture a specific method of aggregating the functionality of multiple adapters. Each composition block class takes a sequence of adapter identifiers plus optional configuration as arguments. The defined adapter setup is then parsed at runtime by Adapters to allow for dynamic switching between adapters per forward pass. Above the different composition blocks are illustrated. A composition could look as follows:

```python
config = "mam" # mix-and-match adapters

model.add_adapter("a", config=config)
model.add_adapter("b", config=config)
model.add_adapter("c", config=config)

model.set_active_adapters(Stack("a", Parallel("b", "c")))

print(model.active_adapters) # The active congif is: Stack[a, Parallel[b, c]]
```

For details also check out [this blog post](https://adapterhub.ml/blog/2021/04/version-2-of-adapterhub-released/) and the [documentation](https://docs.adapterhub.ml/adapter_composition.html). 

## Evaluating Adapter Performance

![](/static/images/eval_results.png "Performance of different adapter architectures overdiffernt tasks evaluated with the RoBERTa model." )

In addition to the ease of use aforementioned, we show that the adapter methods offered by our library are performant across a range of settings. To this end, we conduct evaluations on the single adapter implementations made available by Adapters.

The obvious takeaway from our evaluations is that all adapter implementations offered by our framework are competitive with full model fine-tuning, across all task classes. Approaches that offer more tunable hyper-parameters (and thus allow for easy scaling) such as Bottleneck adapters, LoRA, and Prefix Tuning predictably have the highest topline performance, often surpassing full fine-tuning. However, extremely parameter-frugal methods like (IA) 3 , which add < 0.005% of the parameters of the base model, also perform commendably and only fall short by a small fraction. Finally, the Compacter is the least volatile among the single methods, obtaining the lowest standard deviation between runs on the majority of tasks.


## References
- He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2021, October). Towards a Unified View of Parameter-Efficient Transfer Learning. In International Conference on Learning Representations.
- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q.D., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. ICML.
- Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021, October). LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.
- Li, X. L., & Liang, P. (2021, August). Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 4582-4597).
- Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. A. (2022). Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35, 1950-1965.
- Mao, Y., Mathias, L., Hou, R., Almahairi, A., Ma, H., Han, J., ... & Khabsa, M. (2022, May). UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 6253-6264).
- Pfeiffer, J., Vulic, I., Gurevych, I., & Ruder, S. (2020). MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer. ArXiv, abs/2005.00052.
- Pfeiffer, J., Ruder, S., Vulic, I., & Ponti, E. (2023). Modular Deep Learning. ArXiv, abs/2302.11529.
- Zhou, H., Wan, X., Vulić, I., & Korhonen, A. (2023). AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning. arXiv preprint arXiv:2301.12132.