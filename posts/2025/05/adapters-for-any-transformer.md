---
title: "Adapters for Any Transformer On the HuggingFace Hub"
date: 2025-05-21
authors:
  - name: The AdapterHub Team
summary: |
  The latest release of Adapters v1.2.0 introduces a new adapter plugin interface that enables adding adapter functionality to nearly any Transformer model.
  We go through the details of working with this interface and various additional novelties of the library.
---

In recent weeks and months, we've been working on greatly improving the integration of the _Adapters_ library with the Hugging Face ecosystem.
This has resulted in our [new adapter plugin interface](https://docs.adapterhub.ml/plugin_interface.html).
The plugin interface allows you to integrate most of the _Adapters_ library's features into nearly any Transformers model on the Hugging Face Hub with minimal effort.
In this post, we'll walk you through using the plugin interface step by step and also show what else is new in the _Adapters_ library.

[TOC]

You can find _Adapters_ [on GitHub](https://github.com/Adapter-Hub/adapters) or install it via pip:

```bash
pip install -U adapters
```

## Adapters for Any Transformer with Plugin Interface

_As notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_Interface_Qwen.ipynb)_

In the following, we'll walk through adding adapter support to a custom or not pre-supported model with the _Adapters_ library's [plugin interface](https://docs.adapterhub.ml/plugin_interface.html). Specifically, we'll be writing a plugin interface for the Qwen 3 model and then train an adapter for mathematical reasoning.

**Important:** The interface below for Qwen 2 and Qwen 3 already comes pre-supported in _Adapters_, so you could skip this section entirely! It's merely to showcase how you could define interfaces for your own custom models!
You can find a list of all pre-supported models [in our docs](https://docs.adapterhub.ml/model_overview.html).

### Understanding the Model Architecture

Before creating our plugin interface, let's understand the basic structure of Qwen 3:

- Like most Transformer language models, it consists of an embedding layer followed by a series of decoder layers
- Each layer contains a self-attention block and an MLP block
- The self-attention block includes query, key, value, and output projections
- The MLP block includes multiple linear projections
- Qwen applies layer norms *before* the self-attention and MLP blocks

To create an adapter interface, we need to map these components to the appropriate adapter hooks.

### Creating the Plugin Interface

Now we'll create a plugin interface for Qwen 3 that maps the model's architecture to the adapter framework.


```python
import adapters
from adapters import AdapterModelInterface
from transformers import AutoModelForCausalLM

plugin_interface = AdapterModelInterface(
    # Specify which adapter methods to enable
    adapter_methods=["lora", "reft", "bottleneck"],
    
    # Map the model's components to the adapter interface
    model_embeddings="embed_tokens",      # Embedding layer
    model_layers="layers",                # Transformer layers
    layer_self_attn="self_attn",          # Self-attention module in each layer
    layer_cross_attn=None,                # Qwen doesn't have cross-attention
    
    # Projection matrices within the attention module
    attn_k_proj="k_proj",                 # Key projection
    attn_q_proj="q_proj",                 # Query projection
    attn_v_proj="v_proj",                 # Value projection
    attn_o_proj="o_proj",                 # Output projection
    
    # MLP projections
    layer_intermediate_proj="mlp.up_proj",  # Up projection in MLP
    layer_output_proj="mlp.down_proj",      # Downward projection in MLP

    layer_pre_self_attn="input_layernorm",  # Hook directly before self-attention
    layer_pre_ffn="post_attention_layernorm",  # Hook directly before MLP
    # Qwen applies layer norms before attention and MLP, so no need to add them here
    layer_ln_1=None,
    layer_ln_2=None,
)
```

Each parameter in the interface maps to specific module names in the model's architecture, allowing the adapter methods to hook into the right components.

### Loading the Model and Initializing with the Interface

Now, let's load the Qwen 3 model and initialize it with our plugin interface.


```python
# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B-Base",  # Using the 1.7B version
    device_map="auto",  # Automatically distribute model across available GPUs
    torch_dtype="bfloat16",  # Use half-precision for faster computation
)
```


```python
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

# Set the pad token ID to be different from the model's EOS token
tokenizer.pad_token_id = 151645
model.config.pad_token_id = tokenizer.pad_token_id
```


```python
# Initialize the adapter framework with our plugin interface
# Remove the interface argument to use the default interface
adapters.init(model, interface=plugin_interface)
```

### Adding and Training an Adapter

With the interface in place, we can now add an adapter to our model.
In this example, we'll train a [bottleneck adapter](https://docs.adapterhub.ml/methods.html#bottleneck-adapters). You can easily switch to [one of the other supported adapter methods](https://docs.adapterhub.ml/overview.html#table-of-adapter-methods) (e.g. LoRA) by changing the `adapter_config`.


```python
from adapters import SeqBnConfig, LoRAConfig

# Add a LoRA adapter
adapter_name = "qwen-math-adapter"
adapter_config = SeqBnConfig(
    reduction_factor=32,  # Bottleneck size
)
# Alternatively e.g.: 
# adapter_config = LoRAConfig(
#     r=32,  # Rank of the low-rank decomposition
#     alpha=16,  # Scaling factor for LoRA
# )

model.add_adapter(adapter_name, config=adapter_config)

# Activate the adapter
model.set_active_adapters(adapter_name)

# Set the model to train only the adapter parameters
model.train_adapter(adapter_name)

# Verify adapter was correctly added
print(model.adapter_summary())
```

### Loading & Processing the GSM8K Dataset for Fine-tuning

For this example, we'll use the GSM8K dataset to fine-tune our model for solving grade school math problems.


```python
from datasets import load_dataset

# Load the GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main")
print(dataset)
```


```python
# Explore sample data
print("Sample question:")
print(dataset["train"][0]["question"])
print("\nSample answer:")
print(dataset["train"][0]["answer"])
```

We need to tokenize our math problems and their solutions for training.


```python
def preprocess_function(examples):
    # Create full prompts with question and expected answer format
    prompts = [
        f"Solve the following math problem step-by-step:\n\nQuestion: {q}\n\nAnswer: {a} <|endoftext|>"
        for q, a in zip(examples["question"], examples["answer"])
    ]
    
    # Tokenize as regular sequences
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=2048)
    
    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

print("Dataset processed!")
```

### Fine-tuning the Adapter

Now we can fine-tune our adapter for solving math problems.


```python
from transformers import TrainingArguments
import numpy as np


# Set up training arguments
training_args = TrainingArguments(
    output_dir="./qwen-math-adapter",
    per_device_train_batch_size=2,  # Increase or decrease based on GPU memory
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    num_train_epochs=1,  # More epochs for complex task
    save_steps=30,
    eval_steps=30,
    logging_steps=10,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="loss",  # Use loss as metric for best model
    greater_is_better=False,  # Lower loss is better
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch sizes
    bf16=True,  # Use mixed precision
)
```


```python
# Split dataset into train and validation
# Use a bugger/ smaller subset as needed
train_dataset = tokenized_dataset["train"].select(range(min(len(tokenized_dataset["train"]), 4000)))
eval_dataset = tokenized_dataset["test"].select(range(min(len(tokenized_dataset["test"]), 200)))

print(f"Training on {len(train_dataset)} examples and evaluating on {len(eval_dataset)} examples")
```


```python
from adapters import AdapterTrainer
from trl import DataCollatorForCompletionOnlyLM

# Initialize the trainer
trainer = AdapterTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    data_collator=DataCollatorForCompletionOnlyLM(response_template="Answer:", tokenizer=tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train only the adapter parameters
trainer.train()
```

After training, we can save just the adapter weights.

```python
# Save only the adapter weights
model.save_adapter("./qwen-math-adapter", adapter_name)
```

Additionally, we can push our newly trained adapter to the Hugging Face Model Hub:

```python
model.push_adapter_to_hub("qwen-math-adapter", adapter_name)
```

## Multi-Task Learning with Adapters

The _Adapters_ library has long supported multi-task learning methods such as [AdapterFusion](https://docs.adapterhub.ml/adapter_composition.html#fuse).
In v1.2.0, MTL-LoRA has been added as a new multi-task method for adapters.

MTL-LoRA was introduced in "MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning" ([Yang et al., 2024](https://arxiv.org/pdf/2410.09437)) and enhances LoRA for multi-task learning (MTL) by improving task differentiation and knowledge sharing.
It introduces a task-specific low-rank learnable matrix $\Lambda_t$ to better capture task-specific information and utilizes $n$ low-rank up-projection matrices for diverse information-sharing. A weighted averaging mechanism integrates these matrices, allowing adaptive knowledge transfer across tasks. Specifically, the MTL-LoRA output for task $t$ is formulated as:  

$$
h_t = (W + \Delta W_t)x_t = Wx_t + \sum_{i=1}^n\frac{\text{exp}(w_t^i/\tau)B^i}{\sum_{j=1}^n\text{exp}(w_t^{j}/\tau)}\Lambda_t A x_t
$$

where $\tau$ controls the sharpness of weight distribution. 

`MTL-LoRA` is trainable with `MultiTask` composition and a datasets wich contains `task_ids` column (see. [`MultiTask` Composition](https://docs.adapterhub.ml/adapter_composition.html#multitask)).


_Example_:
```python
from adapters import MTLLoRAConfig
import adapters.composition as ac

config = MTLLoRAConfig(
    r=8,
    alpha=16,
    n_up_projection=3,
)

model.add_adapter("i", config)
model.add_adapter("k", config)
model.add_adapter("l", config)

model.share_parameters(
    adapter_names=["i", "k", "l"],
)

model.active_adapters = ac.MultiTask("i", "k", "l")
```

## New Adapter Method: VeRA

Vera is a LoRA based fine-tuning method proposed by [Kopiczko et al. (2024)](https://arxiv.org/pdf/2310.11454). In Vera, we add frozen matrices A and B that are shared across all layers. It reduces the number of trainable parameters but maintains the same performance when compared to LoRA. Furthermore, trainable scaling vectors $b$ and $d$ are introduced and are multipled by the frozen matrices to result in the equation:

$$ h = W_{0}x + \Lambda_{b}B\Lambda_{d}Ax $$

where $\Lambda_{b}$ and $\Lambda_{d}$ receive updates during training.

_Example_:
```python
from adapters import VeraConfig

config = VeraConfig()
model.add_adapter("vera_config", config=config)
```

## Summary

The latest Adapters library release introduces a powerful plugin interface that allows extending adapter functionality to virtually any Transformer model on the HuggingFace Hub with minimal effort.
This release also brings new multi-task learning capabilities through MTL-LoRA, and adds the parameter-efficient VeRA adapter method.
For the full list of changes, refer to [the release notes of v1.2.0](https://github.com/adapter-hub/adapters/releases/tag/v1.2.0).

## Citation
If you use _Adapters_ in your research, please cite:

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
