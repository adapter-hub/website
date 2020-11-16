---
title: Adapting Transformers with AdapterHub
date: 2020-11-17
author:
  name: Clifton Poth
  twitter: "@clifapt"
summary: |
  Adapters are a new, efficient and composable alternative to full fine-tuning of pre-trained language models.
  AdapterHub makes working with adapters accessible by providing a framework for training, sharing, discovering and consuming adapter modules.
  This post provides an extensive overview.
paper:
    citation: "Pfeiffer, J., Rücklé, A., Poth, C., Kamath, A., Vulic, I., Ruder, S., Cho, K., & Gurevych, I. (2020). AdapterHub: A Framework for Adapting Transformers. ArXiv, abs/2007.07779."
    url: "https://arxiv.org/pdf/2007.07779.pdf"
---

Pre-trained transformers have led to considerable advances in NLP, achieving state-of-the-art results across the board. 
Models such as BERT ([Devlin et al., 2019](https://arxiv.org/pdf/1810.04805.pdf)) and RoBERTa ([Liu et al., 2019](https://arxiv.org/pdf/1907.11692.pdf)) several millions of parameters, and thus, sharing and distributing fine-tuned transformer models can be prohibitive. 

**Adapters**, small layers inserted into every layer of a Transformer-based language model, recently have been introduced as a promising alternative to full fine-tuning of pre-trained models.
Adapters overcome various issues with the established full fine-tuning approach:
they are **parameter-efficient**, **speed up training iterations** and they are especially **shareable** and **composable** due to their modularity and compact size.
Most importantly, they often perform **on-par with state-of-the-art full fine-tuning**.

We have developed **[AdapterHub, A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779.pdf)** which makes working with adapters easily accessible by integrating them with [Huggingface's Transformers](https://github.com/huggingface/transformers), a popular framework for transformer-based language models.
In the following, we will go through the process of training, sharing, consuming and composing adapters with AdapterHub.


## A Short Introduction to Adapters

![](/static/images/steps.gif "Steps of working with adapters")

Adapters provide a lightweight alternative to fully fine-tuning a pre-trained language model on a downstream task.
For a transformer-based architecture, a small set of new parameters is introduced in every transformer layer.
While different adapter architectures are possible, a simple layout using a down- and an up-projection layer first introduced by [Houlsby et al. (2020)](https://arxiv.org/pdf/1902.00751.pdf) has proven to work well (see Figure 1 for illustration).
In many cases, adapters perform on-par with fully fine-tuned models.

During training on the target task, all weights in the pre-trained language model are kept fix.
The only weights to be updated are those introduced by the adapter modules.
This results in modular knowledge representations which subsequently can be easily extracted from the underlying language model.
The extracted adapter modules then can be distributed independently and plugged in into a language model dynamically.
The encapsulated character of adapters also allows for easy exchange and composition of different adapters ([Pfeiffer et al., 2020a](https://arxiv.org/pdf/2005.00247.pdf)).
Since this workflow of using adapters is very universal, it can potentially be applied to a wide range of different use cases.
As an example, adapters have been used successfully for zero-shot cross-lingual transfer between different tasks ([Pfeiffer et al., 2020b](https://arxiv.org/pdf/2005.00052.pdf)).
Figure 1 illustrates the described adapter workflow.

![](/static/images/size_comparison.png "Size comparison of a fully fine-tuned model and an adapter")

Using adapters provides various benefits, especially in parameter efficiency.
With adapter fine-tuning, the amount of updated parameters makes only about 1% of fully fine-tuning a language model in many cases, often just a few Megabytes.
This makes it easy to share adapters, store adapters for many different tasks and load additional adapters on-the-fly.
Additionally, their compact size make adapters a computationally efficient fine-tuning choice ([Rücklé et al., 2020](https://arxiv.org/pdf/2010.11918.pdf)).


## What is AdapterHub?

With AdapterHub, we have developed a framework which makes working with adapters straightforward.
AdapterHub is divided into two core components: [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers), a library built on top of HuggingFace `transformers` that integrates adapter support into various popular Transformer-based language models, and [the Hub](https://adapterhun.ml/explore), an open platform for sharing, exploring and consuming pre-trained adapters.

![](/static/images/lifecycle.png "The AdapterHub lifecycle")

Based on Figure 3, we'll go through the lifecycle of working with AdapterHub on a high level:
HuggingFace `transformers` (🤗) builds the backbone of our framework.
A user who wants to train an adapter (👩🏾‍💻) loads a pre-trained language model (🤖) from 🤗.
In ①, new adapter modules are introduced to the loaded language model.
Afterwards, 👩🏾‍💻 trains the adapter on a downstream task (②).
As soon as training has completed, 👩🏾‍💻 can extract the trained adapter weights from the (unaltered) 🤖 in ③.
👩🏾‍💻 packs the adapter weights and uploads them to the Hub.
Here, 👨🏼‍💻 can find the pre-trained adapter in step ④.
Together with downloading the matching 🤖 from 🤗, 👨🏼‍💻 then can download the adapter from the Hub and integrate it into his own model (⑤).
In ⑥, he lastly can apply 👩🏾‍💻's adapter for his own purposes.

In the following, we will have a look at some of this steps in a bit more detail.

## Training an Adapter

<a href="https://colab.research.google.com/drive/1QR2Vy4mJFUi5r3HaQVROY3dQ9QMTJqhR?usp=sharing" target="_blank">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Training an adapter on a downstream task is a straightforward process using `adapter-transformers` which can be installed via pip:

```bash
pip install adapter-transformers
```

This package is fully compatible with HuggingFace's `transformers` library and can act as a drop-in replacement. Therefore, we can instantiate a pre-trained language model and tokenizer in the familiar way:

```python
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base"
)
config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"},
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)
```

There is one difference compared to HuggingFace transformers in the code above:
We use the new class `RobertaModelWithHeads` which allows a more flexible way of configuring prediction heads.

The next steps configure our adapter setup. Note that these are the only lines additionally needed to switch from full fine-tuning to adapter training.

```python
from transformers import AdapterType

# Add a new adapter
model.add_adapter("rotten_tomatoes", AdapterType.text_task)
# Add a matching classification head
model.add_classification_head("rotten_tomatoes", num_labels=2)
# Activate the adapter
model.train_adapter("rotten_tomatoes")
```

We add a new adapter to our model by calling `add_adapter()`. We pass a name (`"rotten_tomatoes"`) and [the type of adapter](https://docs.adapterhub.ml/adapters.html#adapter-types) (task adapter). Next, we add a binary classification head. It's convenient to give the prediction head the same name as the adapter. This allows us to activate both together in the next step. The `train_adapter()` method does two things:

1. It freezes all weights of the pre-trained model so only the adapter weights are updated during training.
2. It activates the adapter and the prediction head such that both are used in every forward pass.

All the rest of the training process is identical to a full fine-tuning approach. Check out [the Colab notebook on adapter training](https://colab.research.google.com/drive/1QR2Vy4mJFUi5r3HaQVROY3dQ9QMTJqhR?usp=sharing) to see the full code.

In the end, the trained adapter can be exported to the file system using a single line of code:

```python
model.save_adapter("./final_adapter", "rotten_tomatoes")
```

## Interacting with the Hub

<a href="https://colab.research.google.com/drive/1ovA1_ENGU1TT4T6nz2bW2bzq8-Lg8mMW?usp=sharing" target="_blank">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The adapter weights trained in the previous section subsequently can be distributed via the Hub, the second core component of the AdapterHub framework.
The Hub infrastructure is based on plain YAML description files contributed to a central GitHub repository.
The full process of contributing pre-trained adapters is described [in our documentation](https://docs.adapterhub.ml/contributing.html).

The [Explore section](https://adapterhub.ml/explore) of the AdapterHub website acts as the starting point for discovering and consuming available pre-trained adapters. A matching adapter can be selected by task domain, training dataset, model architecture and adapter architecture and loaded into `adapter-transformers` in the following.

Before loading the adapter, we instantiate the model we want to use, a pre-trained `bert-base-uncased` model from HuggingFace.

```python
from transformers import AutoTokenizer, AutoModelWithHeads

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
```

Using `load_adapter()`, we download and add a pre-trained adapter from the Hub. The first parameter specifies the name of the adapter whereas the second selects the [adapter architectures](https://docs.adapterhub.ml/adapters.html#adapter-architectures) to search for.

Also note that most adapters come with a prediction head included. Thus, this method will also load the question answering head trained together with the adapter.

```python
adapter_name = model.load_adapter("qa/squad1@ukp", config="houlsby")
```

With `set_active_adapters()` we tell our model to use the adapter we just loaded in every forward pass.

```python
model.set_active_adapters(adapter_name)
```

Again, these are all changes needed to set up a pre-trained language model with a pre-trained adapter.
The rest of the inference is identical to a setup without adapters.
To see a full example, check out [the Colab notebook for adapter inference](https://colab.research.google.com/drive/1ovA1_ENGU1TT4T6nz2bW2bzq8-Lg8mMW?usp=sharing).

## Adapter Composition

<a href="https://colab.research.google.com/drive/1bt_EmBe00s4TldihSavA7ha9Pq2inDY4?usp=sharing" target="_blank">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

As presented earlier, adapters are especially suitable for various kinds of compositions on a new target task.
One of these composition approaches is _AdapterFusion_ ([Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00247.pdf)) which is also tightly integrated into AdapterHub.

The knowledge learned by multiple pre-trained adapters from the Hub can be leveraged to solve a new target task.
In this setup, only a newly introduced fusion layer is trained while the rest of the model is kept fix.

First, we load three adapters pre-trained on different tasks from the Hub: MultiNLI, QQP and QNLI. As we don't need their prediction heads, we pass `with_head=False` to the loading method. Next, we add a new fusion layer that combines all the adapters we've just loaded. Finally, we add a new classification head for our target task on top.

```python
from transformers import AdapterType

# Load the pre-trained adapters we want to fuse
model.load_adapter("nli/multinli@ukp", AdapterType.text_task, load_as="multinli", with_head=False)
model.load_adapter("sts/qqp@ukp", AdapterType.text_task, with_head=False)
model.load_adapter("nli/qnli@ukp", AdapterType.text_task, with_head=False)
# Add a fusion layer for all loaded adapters
model.add_fusion(["multinli", "qqp", "qnli"])

# Add a classification head for our target task
model.add_classification_head("cb", num_labels=len(id2label))
```

The last preparation step is to define and activate our adapter setup. Similar to `train_adapter()`, `train_fusion()` does two things: It freezes all weights of the model (including adapters!) except for the fusion layer and classification head. It also activates the given adapter setup to be used in very forward pass.

The syntax for the adapter setup (which is also applied to other methods such as `set_active_adapters()`) works as follows:

- a single string is interpreted as a single adapter
- a list of strings is interpreted as a __stack__ of adapters
- a _nested_ list of strings is interpreted as a __fusion__ of adapters

```python
# Unfreeze and activate fusion setup
adapter_setup = [
  ["multinli", "qqp", "qnli"]
]
model.train_fusion(adapter_setup)
```

See the full training example in [the Colab notebook on AdapterFusion](https://colab.research.google.com/drive/1bt_EmBe00s4TldihSavA7ha9Pq2inDY4?usp=sharing).

## Conclusion

Adapters are a promising new approach to transfer learning in NLP, providing benefits in efficiency and modularity.
AdapterHub provides tools for the full lifecycle of interacting with adapters.
The integration into the successful HuggingFace `transformers` framework makes it straightforward to adapt training setups to adapters.
AdapterHub is continuously evolving with the addition of adapter support to new models, the integration of new application scenarios for adapters and a growing platform of pre-trained adapter modules.

## References

- Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q.D., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. ICML.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. ArXiv, abs/1907.11692.
- Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I. (2020). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. ArXiv, abs/2005.00247.
- Pfeiffer, J., Vulic, I., Gurevych, I., & Ruder, S. (2020). MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer. ArXiv, abs/2005.00052.
- Rücklé, A., Geigle, G., Glockner, M., Beck, T., Pfeiffer, J., Reimers, N., & Gurevych, I. (2020). AdapterDrop: On the Efficiency of Adapters in Transformers. ArXiv, abs/2010.11918.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing. ArXiv, abs/1910.03771.
