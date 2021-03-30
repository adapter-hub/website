---
title: Version 2 of adapter-transformers Released
date: 2020-04-01
author:
  name: Clifton Poth
  twitter: "@clifapt"
summary: |
  Today, version 2 of adapter-transformers has been released.
  adapter-transformers, built on top of HuggingFace's Transformers library, is the heart of the AdapterHub framework that makes working with adapters easy.
  The new version brings new possibilities to compose adapters, also in more complex setups, as well as the support for new Transformers model architectures.
---

Adapters, a light-weight alternative to full fine-tuning of state-of-the-art language models, have enabled new possibilities of composing task-specific knowledge from multiple sources, for example for multi-task transfer learning ([Pfeiffer et al., 2021](https://arxiv.org/pdf/2005.00247.pdf)) or for cross-lingual transfer ([Pfeiffer et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-main.617.pdf)).
One of the great advantages of adapters is their modularity that allows not only mentioned scenarios but also various other composition possibilities.

Today, are realeasing version 2 of `adapter-transformers` which will make it easier than before to take advantage of this composability and flexibility of adapters.
`adapter-transformers`, which is an extension of the great [Transformers library by HuggingFace](https://huggingface.co/transformers/), is the heart of the [AdapterHub framework](https://adapterhub.ml/) which aims to simplify the full lifecycle of working with adapters.
(Check out [our first blog post for more on that](https://adapterhub.ml/blog/2020/11/adapting-transformers-with-adapterhub/).)

In the following few sections, we will have a look into everything new and changed in the v2 release.
You can find `adapter-transformers` [on GitHub](https://github.com/Adapter-Hub/adapter-transformers) or install it via pip:

```bash
pip install -U adapter-transformers
```

## What's new

### Adapter composition blocks

The new version introduces a radically different way to define adapter setups in a Transformers model,
allowing much more advanced and flexible adapter composition possibilities.
An example setup using this new, modular composition mechanism might look like this:

```python
import transformers.adapters.composition as ac

model.active_adapters = ac.Stack("a", ac.Split("b", "c", split_index=60))
```

As we see, the basic building blocks of this setup are simple objects representing different possibilities to combine single adapters.
In the example, `Stack` describes stacking layers of adapters on top of each other,
as used in the _MAD-X_ framework for cross-lingual transfer.
`Split` describes splitting the input sequences between two adapters at a specified index.
Thus, in the shown setup, in each adapter layer, the input is first passed through adapter `a` before being split up between adapters `b` and `c` and passed through both adapters in parallel.

Besides the two blocks shown, `adapter-transformers` currently also includes a `Fuse` block (for [_AdapterFusion_](https://arxiv.org/pdf/2005.00247.pdf)) and a `Parallel` block (see below).
All of these blocks derive from `AdapterCompositionBlock`, and they can be combined in flexibly in many ways.
For more information on specifying the active adapters using `active_adapters` and the new composition blocks,
refer to the [corresponding section in our documentation](adapter_composition.md).

### New model support: Adapters for BART and GPT-2

The two new model architectures added in v2.0, BART and GPT-2, start the process of integrating adapters into sequence-to-sequence models, with more to come.

We have [a separate blog post]() presenting our results when training adapters on both models and new adapters in the Hub.

### AdapterDrop

Version 2 of `adapter-transformers` integrates some of the key ideas presented in _AdapterDrop_ [(Rücklé et al., 2020)](https://arxiv.org/pdf/2010.11918.pdf), namely, (1) parallel multi-task inference and (2) _robust_ AdapterDrop training. 

Parallel multi-task inference, for any given input, runs multiple task adapters in parallel and thereby achieves considerable improvements in inference speed compared to sequentially running multiple BERT models (see our paper for more details). The `Parallel` adapter composition block implements this behavior, which we describe in more detail [here](adapter_composition.html#parallel).

A central advantage of multi-task inference is that it shares the computations in lower transformer layers across all inference tasks (before the first adapter block). Dropping out adapters from lower transformer layers can thus result in even faster inference speeds, but it often comes at the cost of lower accuracies. To allow for _dynamic_ adjustment of the number of dropped adapter layers at run-time regarding the available computational resources, we introduce _robust_ adapter training. This technique drops adapters from a random number of lower transformer layers in each training step. The resulting adapter can be adjusted at run-time regarding the number of dropped layers, to dynamically select between a higher accuracy or faster inference speeds.
We present an example for robust _AdapterDrop_ training [in this Colab notebook](https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/Adapter_Drop_Training.ipynb).


### Transformers upgrade

Version 2.0.0 upgrades the underlying HuggingFace Transformers library from v3.5.1 to v4.2.2, bringing many awesome new features created by HuggingFace.

## What has changed

### Unified handling of all adapter types

_Includes breaking changes ⚠️_

The new version removes the hard distinction between _task_ and _language_ adapters (realized using the `AdapterType` enumeration in v1) everywhere in the library.
Instead, all adapters use the same set of methods.
This, of course, leads to some breaking changes.
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
Nonetheless, rhis does not visibly affect loading older adapters with the new version.
When loading an adapter trained with v1 in a newer version, `adapter-transformers` will automatically convert the weights to the new format.
However, loading adapters trained with newer versions into an earlier v1.x version of the library does not work.

Additionally, there have been some changes in the saved configuration dictionary, also including automatic conversions from older versions.

### Refactorings in adapter implementations

There have been some refactorings mainly in the adapter mixin implementations.
Further details can be found [in the guide for adding adapters to a new model](https://github.com/Adapter-Hub/adapter-transformers/blob/master/adding_adapters_to_a_model.md).

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
