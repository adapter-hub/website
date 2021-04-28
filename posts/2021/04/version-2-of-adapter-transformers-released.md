---
title: Version 2 of adapter-transformers Released
date: 2021-04-29
authors:
  - name: Clifton Poth
    twitter: "@clifapt"
  - name: Hannah Sterz
    twitter: "@Hannah70676760"
summary: Today, we are releasing version 2 of adapter-transformers. This release introduces several exciting new ways for composing adapters through composition blocks, including AdapterFusion, ParallelInference, Adapter stacking, and combinations thereof. Furthermore, we now support new Transformer architectures such as GPT-2 and BART.
---

![](/static/images/v2_blocks.png "Illustration of adapter composition blocks supported in v2 of adapter-transformers.")

Adapters, a light-weight alternative to full language model fine-tuning, enable new ways of composing task-specific knowledge from multiple sources, e.g., for multi-task transfer learning ([Pfeiffer et al., 2021](https://arxiv.org/pdf/2005.00247.pdf)) or cross-lingual transfer ([Pfeiffer et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-main.617.pdf)).
One of the most important advantages of adapters is their modularity, which allows many exciting possibilities for composition beyond the ones mentioned above.

Today, we are releasing version 2 of `adapter-transformers`, which makes it easier to take advantage of the composability and flexibility of adapters.
`adapter-transformers` --- an extension of the great [Transformers library by HuggingFace](https://huggingface.co/transformers/) --- is the heart of the [AdapterHub framework](https://adapterhub.ml/) and simplifies the entire adapter lifecycle.
(Check out [our first blog post for more on this](https://adapterhub.ml/blog/2020/11/adapting-transformers-with-adapterhub/).)

In the following sections, we will discuss all new features and changes that we introduce with the v2 release.
You can find `adapter-transformers` [on GitHub](https://github.com/Adapter-Hub/adapter-transformers) or install it via pip:

```bash
pip install -U adapter-transformers
```

## What's new

### Adapter composition blocks

The new version introduces a radically different way to define adapter setups in a Transformer model,
allowing much more advanced and flexible adapter composition possibilities.
An example setup using this new, modular composition mechanism might look like this:

```python
import transformers.adapters.composition as ac

model.active_adapters = ac.Stack("a", ac.Split("b", "c", split_index=60))
```

As we can see, the basic building blocks of this setup are simple objects representing different possibilities to combine individual adapters.
In the above example, `Stack` describes stacking adapters layers on top of each other,
e.g., as it is used in the _MAD-X_ framework for cross-lingual transfer.
`Split` results in splitting the input sequences between two adapters at a specified `split_index`.
In the depicted setup, at every transformer layer the token representations are first passed through adapter `a` before being split at the `split_index` and passed through adapters `b` and `c` respectively.

Besides the two blocks shown, `adapter-transformers` includes a `Fuse` block (for [_AdapterFusion_](https://arxiv.org/pdf/2005.00247.pdf)) and a `Parallel` block (see below).
All of these blocks are derived from `AdapterCompositionBlock`, and they can be flexibly combined in even very complex scenarios.
Figure 1 shows an illustration of the structure of each composition block.
For more information on specifying the active adapters using `active_adapters` and the new composition blocks,
refer to the [corresponding section in our documentation](https://docs.adapterhub.ml/adapter_composition.html).

### New model support: Adapters for BART and GPT-2

v2 adds support for BART and GPT-2, marking a new type of models we support in the framework, namely sequence-to-sequence models (more to come!)

We have [a separate blog post](https://adapterhub.ml/blog/2021/04/adapters-for-bart-and-gpt2/) that studies the effectiveness of adapters within these two models in greater detail! This blog post also includes a hands-on example where we train GPT-2 to generate poetry.

### AdapterDrop

Version 2 of `adapter-transformers` integrates some of the key ideas presented in _AdapterDrop_ [(Rücklé et al., 2020)](https://arxiv.org/pdf/2010.11918.pdf), namely, (1) parallel multi-task inference and (2) _robust_ AdapterDrop training. 

Parallel multi-task inference, for any given input, runs multiple task adapters in parallel and thereby achieves considerable improvements in inference speed compared to sequentially running multiple Transformer models (see the paper for more details). The `Parallel` adapter composition block implements this behavior, which we describe in more detail [here](adapter_composition.html#parallel).

A central advantage of multi-task inference is that it shares the computations in lower transformer layers across all inference tasks (before the first adapter block). Dropping out adapters from lower transformer layers can thus result in even faster inference speeds, but it often comes at the cost of lower accuracies. To allow for _dynamic_ adjustment of the number of dropped adapter layers at run-time regarding the available computational resources, we introduce _robust_ adapter training. This technique drops adapters from a random number of lower transformer layers in each training step. The resulting adapter can be adjusted at run-time regarding the number of dropped layers, to dynamically select between a higher accuracy or faster inference speeds.
We present an example for robust _AdapterDrop_ training [in this Colab notebook](https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/Adapter_Drop_Training.ipynb).


### Transformers upgrade

Version 2.0.0 upgrades the underlying HuggingFace Transformers library from v3.5.1 to v4.2.2, bringing many awesome new features created by HuggingFace.

## What has changed

### Unified handling of all adapter types

_Includes breaking changes ⚠️_

The new version removes the hard distinction between _task_ and _language_ adapters (realized using the `AdapterType` enumeration in v1) everywhere in the library.
Instead, all adapters use the same set of methods.
This results in some breaking changes.
For example, you don't have to specify the adapter type anymore when adding a new adapter.
Instead of...
```python
# OLD (v1)
model.add_adapter("name", AdapterType.text_task, config="houlsby")
```
... you would simply write...
```python
# NEW (v2)
model.add_adapter("name", config="houlsby")
```

A similar change applies for loading adapters from the Hub using `load_adapter()`.

In v1, adapters of type `text_lang` automatically had invertible adapter modules added.
As this type distinction is now removed, adding invertible adapters can be specified via the adapter config.
For example...

```python
# OLD (v1)
model.add_adapter("name", AdapterType.text_task, config="pfeiffer")
```
... in v1 would be equivalent to the following in v2:
```python
# NEW (v2)
model.add_adapter("name", config="pfeiffer+inv")
```

### Removal of `adapter_names` parameter in model forward()

_Includes breaking changes ⚠️_

In v1, it was possible to specify the active adapters using the `adapter_names` parameter in each call to the model's `forward()` method.
With the integration of the new, unified mechanism for specifying adapter setups using composition blocks, this parameter was dropped.
The active adapters now are exclusively set via `set_active_adapters()` or the `active_adapters` property.
For example...

```python
# OLD (v1)
model(**input_data, adapter_names="awesome_adapter")
```
... would become...
```python
# NEW (v2)
model.active_adapters = "awesome_adapter"
model(**input_data)
```

## Internal changes

### Changes to adapter weights dictionaries and config

_Includes breaking changes ⚠️_

With the unification of different adapter types and other internal refactorings, the names of the modules holding the adapters have changed.
This affects the weights dictionaries exported by `save_adapter()`, making the adapters incompatible _in name_.
Nonetheless, this does not visibly affect loading older adapters with the new version.
When loading an adapter trained with v1 in a newer version, `adapter-transformers` will automatically convert the weights to the new format.
However, loading adapters trained with newer versions into an earlier v1.x version of the library does not work.

Additionally, there have been some changes in the saved configuration dictionary, also including automatic conversions from older versions.

### Refactorings in adapter implementations

There have been some refactorings mainly in the adapter mixin implementations.
Most importantly, all adapter-related code has been moved to the `transformers.adapters` namespace.
Further details on the implementation can be found [in the guide for adding adapters to a new model](https://github.com/Adapter-Hub/adapter-transformers/blob/master/adding_adapters_to_a_model.md).

## Conclusion


Version 2 of `adapter-transformers` brings a range of new features to broaden the possibilities of working with adapters.
The library is still under active development, so make sure to check it out [on GitHub](https://github.com/Adapter-Hub/adapter-transformers).
Also, we're always happy for any kind of contributions!

## References

- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q.D., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. ICML.
- Pfeiffer, J., Rücklé, A., Poth, C., Kamath, A., Vulić, I., Ruder, S., Cho, K., & Gurevych, I. (2020). AdapterHub: A Framework for Adapting Transformers. EMNLP.
- Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I. (2020). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. ArXiv, abs/2005.00247.
- Pfeiffer, J., Vulic, I., Gurevych, I., & Ruder, S. (2020). MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer. ArXiv, abs/2005.00052.
- Rücklé, A., Geigle, G., Glockner, M., Beck, T., Pfeiffer, J., Reimers, N., & Gurevych, I. (2020). AdapterDrop: On the Efficiency of Adapters in Transformers. ArXiv, abs/2010.11918.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing. ArXiv, abs/1910.03771.
