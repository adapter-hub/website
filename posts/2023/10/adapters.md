---
title: Introducing Adapters
date: 2023-10-23
authors:
  - name: Clifton Poth, Hannah Sterz, Leon Engländer, Timo Imhof
    twitter: "@clifapt"
summary: |
  Introducing the new Adapters library the new package that supports adding parameter-efficient fine-tuning methods on top of transformers models and composition to achieve modular setups.
paper:
    citation: 
    url: 
---

We are happy to announce the new **Adapters** library. It provides the functionality to add, train, and compose parameter-efficient fine-tuning methods. This blog post summarizes the most important aspects as they are mentioned in the corresponding paper. In times of LLMs with an increasing number of parameters parameter-efficient fine-tuning methods, which do not fine-tune the whole model but instead only update a small number of parameters are promising methods. In addition to the parameter-efficient aspects, the adapters have achieved great results in modular transfer learning. Setups like MAD-X (Pfeiffer et el., 2020) exploit the modular nature of adapters by training separate task and language adapters. Swapping the language adapter during inference allows the transfer to other languages. 

![](/static/images/comparison.png "Version comparison")

**Adapters** aims at providing unifying parameter-efficient and modular transfer learning. It is a self-contained library supporting 10 adapter methods and complex configurations. The adapters can be combined to achieve modular transfer learning with 6 different composition blocks. This provides substantial improvements compared to the predecessor `adapter-transformers:`

1. Decoupled from the HuggingFace `transformers` library
2. Support of 10 Adapter Methods
3. Support of 6 Composition Blocks

## Transformers Integration

The **Adapters** library is self-contained. As a result, the HuggingFace models need to be attached with the adapter functionality:

```python
from transformers import BertModel
import adapters

model = BertModel.from_pretrained("bert-base-uncased")
adapters.init(model)
```

We recommend using the model classes provided by **Adapters** `XXXAdapterModel`(XXX denotes the model architecture e.g. Bert). They provide the adapter functionality without further initialization and support multiple heads, which is relevant when using composition blocks which can handle multiple outputs, for instance, the BatchSplit composition block.

```python
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("roberta-base")
model.add_adapter("adapter1", config="seq_bn") # add the new adapter to the model 
model.train_adapters("adapter1") # freeze the model weights and activate the adapter
```

## Adapter Methods

Each adapter method is defined by a configuration object or string which allows flexible customization of various properties of an adapter module, including placement, capacity, residual connections, initialization etc. We distinguish between single methods consisting of one type of adapter module and complex methods consisting of multiple different adapter module types.

```python
model.add_adapter("seq_bn", config="seq_bn")
model.add_adapter("compacter", config="compacter")
model.add_adapter("prefix_tuning", config="prefix_tuning")
model.add_adapter("lora", config="lora")
model.add_adapter("ia3", config="ia3")
```

### Single Methods

*Adapters* supports single adapter methods that introduce parameters in new feed-forward modules such as bottleneck adapters (Houlsby et al., 2019), introduce prompts at different locations such as prefix tuning (Li and Liang, 2021), reparameterize existing modules such as LoRA (Hu et al., 2022) or re-scale their output representations such as (IA)³ (Liu et al., 2022) (For more information see our [documentation](https://docs.adapterhub.ml/methods.html).)

### Complex Methods

While different efficient fine-tuning methods and configurations have often been proposed as standalone, combining them for joint training has proven to be beneficial (He et al., 2022; Mao et al., 2022). To make this process easier, Adapters provides the possibility to group multiple configuration instances using the ConfigUnion class. This flexible mechanism allows easy integration of multiple complex methods proposed in the literature (as the two examples outlined below), as well as the construction of other, new complex configurations currently not available nor benchmarked in the literature (Zhou et al., 2023).

**Mix-and-Match Adapters** (He et al., 2022) was proposed as a combination of Prefix-Tuning and parallel bottleneck adapters. Using ConfigUnion, this method can be defined as:

```python
config = ConfigUnion( 
	PrefixTuningConfig(bottleneck_size=800), 
	ParallelConfig(), 
) 
model.add_adapter("name", config=config)
```

**UniPELT** (Mao et al., 2022) combines LoRA, Prefix Tuning, and bottleneck adapters in a single unified setup. It additionally introduces a gating mechanism that controls the activation of the different adapter modules.

$$
G m ← σ(W G m · x).
$$

## Modularity

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

## Adapter Evaluation

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
- Zhou, H., Wan, X., Vulić, I., & Korhonen, A. (2023). AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning. arXiv preprint arXiv:2301.12132.