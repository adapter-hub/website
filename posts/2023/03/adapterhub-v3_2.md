---
title: Updates in Adapter-Transformers v3.2
date: 2023-03-03
authors:
  - name: Hannah Sterz
    twitter: "@h_sterz"
  - name: Clifton Poth
    twitter: "@clifapt"
  - name: Leon Engländer
summary: With the newest release of our adapter-transformers library, version 3.2, we add composition blocks for prefix tuning and adapters to several new models.
---


Throughout the last months, we worked on improving the `adapter-transformers` library and including new features. This includes support for new models like CLIP and BEiT, more flexible adapter configuration, and adapter composition for prefix-tuning. In the following, we describe the new features and updates in more detail.

You can find version 3.2 of `adapter-transformers` [on GitHub](https://github.com/Adapter-Hub/adapter-transformers) or install it via pip:

```bash
pip install -U adapter-transformers
```

## Support for adapter configuration strings 
For running experiments at a large scale with varying hyperparameters, it can be annoying to set the correct hyperparameters whenever running the scripts. Now, you can configure the adapter with a string. In previous versions, it was possible to use one of the predefined configurations via a string e.g. `pfeiffer`. From v.3.2 on it is possible to adapt parameters within the string as well.
To create a Pfeiffer adapter with reduction factor 16 you can now use `pfeiffer[bottleneck_size=800]`. This can also help run the example scripts. [Learn more](https://docs.adapterhub.ml/overview.html#configuration-strings)

## Adapter Composition for Prefix Tuning 

![](/static/images/v3_2_prefix_stack.png "Illustration of composition for prefix tuning (Pfeiffer et al.)")
 
Parameter-effifient fine-tuning methods have proven to be modular. Combining multiple adapters can be beneficial for transfer learning across languages. In v.3.2 we add `Stack`, `Parallel` & `BatchSplit` compositions to prefix tuning.
In previous `adapter-transformers` versions, you could combine multiple bottleneck adapters. You could use them in parallel or stack them. Now, this is also possible for prefix tuning adapters. Add multiple prefixes to the same model to combine the functionality of multiple adapters (`Stack`) or perform several tasks simultaneously (`Parallel`, `BatchSplit`). [Learn more](https://docs.adapterhub.ml/adapter_composition.html#stack)

## Enable parallel sequence generation with adapters 
In v3.2 you can use the `Parallel` block in combination with the `model.generate()` method. This allows to generate text for multiple adapters simultaneously. As a result, generation can now be used in a multi task inference setup and generate text for multiple tasks within one forward pass. 

## New model integrations
The new v3.2 of `adapter-transformers` adds support for adapters for several new models: 

- BEiT 
- GPT-J 
- CLIP 
- ALBERT 
- BertGeneration 


## Fixed
- Fixes for GLUE & dependency parsing example script
- Fix access to shared parameters of compacter (e.g. during sequence generation) 
- Fix reference to adapter configs in `T5EncoderModel`
- Fix DeBERTa prefix tuning with enabled relative attention 
- Fix gating for prefix tuning layers 
- Fix input to T5 adapter layers
- Fix AdapterTrainer hyperparameter tuning
- Move loading best adapter to AdapterTrainer class
- Make HuggingFace Hub Mixin work with newer utilities 
- Only compute fusion reg loss if the fusion layer is trained 


### Transformers Update
Version 3.2 of `adapter-transformers` updates the underlying transformers version from v.4.23.1 to v4.26.1

## References

- Pfeiffer, J., Ruder, S., Vulic, I., & Ponti, E. (2023). Modular Deep Learning. ArXiv, abs/2302.11529.
