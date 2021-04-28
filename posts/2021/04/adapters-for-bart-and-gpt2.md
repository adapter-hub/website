---
title: Adapters for BART and GPT2
date: 2021-04-30
authors:
  - name: Hannah Sterz
    twitter: "@Hannah70676760"
  - name: Clifton Poth
    twitter: "@clifapt"
summary: |
  Adapters have proven to be an efficient alternative to fully finetung models. The version 2.0 of the AdapterHub framework includes adapters for the BART and GPT2 models.
---


<p align="center">
<img src="/static/images/BARTLogo.png">
</p>

Adapters are becoming more and more important in machine learning for NLP. They enable us to quickly train and share new task specific models. Adapters are small layers that are stitched into the pre-trained model. During training, only the parameters of the adapter layers are finetuned. Meanwhile, the parameters of the pre-trained model remain frozen. As a result, it is sufficient to store the adapter layers for each task instead of storing fully finetuned models separately for each task. Furthermore, the lower number of parameters requires less memory and makes it easier to share the trained adapters. Adapters also offer possibilities in transfer learning. By using different combinations of adapters the model can be trained on one language and use the adapter on another one. (for more details and examples checkout [this blog post](https://adapterhub.ml/blog/2020/11/adapting-transformers-with-adapterhub/)). [Bapna et al., 2019](https://www.aclweb.org/anthology/D19-1165.pdf) has shown that adapters are useful for sequence to sequence tasks. On a neural machine translation task, they achieved similar results with adapters as with a fully finetuned model.

The AdapterHub framework makes adapters easy to use. Up until now, the framework included adapters for the models BERT, RoBERTa, XML-RoBERTa and DistilBERT. In the new version 2.0, the framework provides adapters for the language generation models BART and GPT-2 as well. This allows us to use adapters for sequence to sequence tasks like e.g. summarization.


## Results of BART and GPT-2 with adapters
 Before we dive into more specific tasks we take a look at the performance on the GLUE tasks. We compare the scores of a fully finetuned model with the scores of a model with adapters configured according to [Pfeiffer et al., 2020a](https://arxiv.org/pdf/2005.00247.pdf) and a model with adapters configured according to [Houlsby et al. 2020](https://arxiv.org/pdf/1902.00751.pdf). The GPT-2 model and BART model achieve the following scores:

<table>
<tr>
<th> GPT-2 </th><th> Full </th><th> Pfeiffer </th><th> Houlsby </th>
</tr>
<tr>
<td> RTE </td><td> 65.0 </td><td> 67.1 </td><td> 67.5 </td>
</tr>
<tr>
<td>   MRPC  </td><td> 83.8 </td><td> 83.5 </td><td> 80.4 </td>
</tr>
<tr>
<td>   STS-B </td><td> 86.7 </td><td> 85.3 </td><td> 85.4 </td>
</tr>
<tr>
<td>   CoLA  </td><td> 33.6 </td><td> 43.0 </td><td> 41.2 </td>
</tr>
<tr>
<td>   SST-2 </td><td> 90.0 </td><td> 90.5 </td><td> 90.9 </td>
</tr>
<tr>
<td>   QNLI  </td><td> 87.6 </td><td> 88.2 </td><td> 88.5 </td>
</tr>
<tr>
<td>   MNLI  </td><td> 82.2 </td><td> 81.6 </td><td> 81.7 </td>
</tr>
<tr>
<td>   QQP   </td><td> 88.5 </td><td> 87.1 </td><td> 87.7 </td>
</tr>
</table>


The fully finetuned GPT-2 model is trained for 4 epochs with a learning rate 0f 1e-4. The adapters are trained for 10 epochs with a learning rate of 1e-4.

<table>
<tr>
<th> BART </th><th> Full </th><th> Pfeiffer </th><th> Houlsby </th>
</tr>
<tr>
<td> RTE </td><td> 71.12 </td><td> 69.7 </td><td> 69.1</td>
</tr>
<tr>
<td>   MRPC  </td><td> 87.5</td><td> 86.8 </td><td> 88.2 </td>
</tr>
<tr>
<td>   STS-B </td><td> 89.0 </td><td> 88.1 </td><td> 88.3 </td>
</tr>
<tr>
<td>   CoLA  </td><td> 46.6 </td><td> 46.1 </td><td> 45.6 </td>
</tr>
<tr>
<td>   SST-2 </td><td> 92.7 </td><td> 93.7 </td><td> 93.6 </td>
</tr>
<tr>
<td>   QNLI  </td><td> 91.6 </td><td> 92.2 </td><td> 93.6 </td>
</tr>
<tr>
<td>   MNLI  </td><td> 85.7 </td><td> 85.9 </td><td> 85.9 </td>
</tr>
<tr>
<td>   QQP   </td><td> 89.3 </td><td> 88.4 </td><td> 88.6 </td>
</tr>
</table>

The fully-finetuned model is trained for 3 epochs with a learning rate of 4e-5. The adapters are trained with early stopping for a maximum of 15 epochs with a learning rate of 1e-4.

The results of the adapters are comparable to those of the fully finetuned model. On some tasks like the SST-2 tasks, the adapters achieve a higher score than the fully finetuned model for GPT-2 and BART. This matches the results of other models with adapters. In general, adapters can be used instead of fully finetuning the model without decreasing results. 

Now we take a look at the scores the adapters can achieve on sequence to sequence tasks. We train the GPT-2 model on the task proposed in [Chen et al., 2020](https://arxiv.org/abs/2004.10404). The task requires the model to learn to generate sentences that can be logically entailed to given data. As an example, the model could be given a table containing the release dates for an album and then the model is given templates it is supposed to fill the blanks in.

> Template: [ENT] was released in 6 [ENT] in [ENT].
> 
> Gold sentence: Black Ice was released in 6 Countries in 2008.

The model can not just enter a number from the table but it needs to count all countries the album was released in 2008. We trained the GPT-2 model with small-sized GPT-2 vocabulary using maximum likelihood estimation. The results are recorded in the following table:
<table>
<tr>
<th></th><th> BLEU-1 </th><th> BLEU-2 </th><th> BLEU-3 </th><th> Adv-Acc </th>
</tr>
<tr>
<td> GPT-2  </td><td> 48.8 </td><td> 27.1 </td><td> 12.6 </td><td> 62.3 </td>
</tr><tr>
<td> GPT-2 + Pfeiffer </td><td> 46.3 </td><td> 24.8 </td><td> 11.2 </td><td> 60.1 </td>
</tr><tr>
<td> GPT-2 + Houlsby </td><td> 45.5 </td><td> 23.9 </td><td> 10.5 </td><td> 59.7 </td>
</tr>
</table>
The models with adapters have a lower score than the fully finetuned model. The adapters might produce slightly lower scores for sequence to sequence tasks than fully finetuning. But the results are close and adapters have several advantages over fully finetuning e.g. shorter training, they need less memory to be stored and they can easily be shared.

To test the BART model on sequence to sequence tasks we evaluated the model on the CNN/Daily Mail dataset ([See et al., 2017](https://arxiv.org/pdf/1704.04368.pdf) [Hermann et al., 2015](https://arxiv.org/pdf/1506.03340.pdf)) and the XSum dataset ([Narayan et al., 2018](https://arxiv.org/pdf/1808.08745.pdf)). Both tasks train the model to summarize newspaper articles. The main difference is that XSum or extreme summary dataset trains the model to output short one sentence summaries. The results of the fully finetuned BART model and the adapters are as follows:
<table>
<tr>
<th></th><th> R1 </th><th> R2 </th><th> RL </th>
</tr><tr>
<td> CNN/Daily mail </td><td> 44.16 </td><td> 21.28 </td><td> 40.90 </td>
</tr><tr>
<td>CNN/Daily mail + Pfeiffer </td><td> 43.40 </td><td> 20.86 </td><td> 30.66 </td>
</tr></table>

<table>
<tr>
<th></th><th> R1 </th><th> R2 </th><th> RL </th>
</tr><tr>
<td> XSum </td><td> 45.14 </td><td> 22.27 </td><td> 37.26 </td>
</tr><tr>
<td> XSum + Pfeiffer </td><td> 43.56 </td><td> 20.56 </td><td> 35.56 </td>
</tr><tr>
<td> XSum + Houlsby </td><td>44.03 </td><td> 20.90 </td><td> 36.01 </td>
</tr></table>

Like the GPT-2 model, the BART model achieves the highest score when it is fully finetuned. The models with adapters achieve slightly lower scores. This further indicates that adapters might achieve a slightly lower score on sequence to sequence tasks in general. But as previously stated they have several advantages over fully finetuning the model.

Version 2.0 of the AdapterHub framework with the addition of adapters for BART and GPT-2 offers new possibilities. Language generation models are better suited for summaries and text generation. Adapters for BART and GPT-2 enable us to tackle these tasks with adapters.

## Hands-on example: Train an adapter to write poems

 [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](
 https://colab.research.google.com/github/hSterz/adapter-transformers/blob/notebooks/notebooks/06_Text_Generation.ipynb) <br>
To give an idea of how adapters can be used for that, we illustrate how to train the GPT-2 model on a poem dataset by [Sheng et al., 2020](https://arxiv.org/pdf/2011.02686.pdf) and let it create its own poems. The dataset contains poems from the Gutenberg project. The full code is available in the corresponding colab notebook. If you have read the previous blog post, this might look very familiar. First, we need to add our adapters.  This is easily done with just a few lines of code:

```python

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# Add a new adapter
model.add_adapter("poem")
# Activate the adapter for training
model.model.train_adapter("poem")

```
We have created the GPT-2 model and added an adapter with `add_adapter()`. We only need to pass the name of the adapter `"poem"`. After adding the new adapter. We call `train_adapter()` and pass the name of our adapter. This does two things. Firstly it freezes all parameters of the pre-trained model such that only the parameters of the adapter are updated during training. Secondly, it activates the adapter so that it is used in the forward pass. Next, we can train our model the same way we would without an adapter. In the end, we can save our trained adapter as follows.

```python

model.save_adapter("path/to/adapter", "poem")

```
We call `save_adapter()` and give the path to the directory where the adapter should be saved and the name of the adapter we want to save.
Now that we have our trained adapter, we want to generate some poems and see what it learned. First, we need to make a model with a language modeling head and load our trained adapter. Then we activate the loaded adapter.

```python

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.load_adapter("path/to/adapter")
model.set_active_adapters("poem")
```

With `load_adapter()` we can load an adapter from the Hub by passing the name of the adapter specified in the hub. We can also load a local adapter by giving the path to the adapter. Then, we activate our adapter such that is used in the forward pass with `set_active_adapters()`.
Finally, we can think of a beginning for a poem and let the model finish it. In this case, the model generates 5 poems for the given beginning. We can choose the one we like most from those. We choose to start our poem with "In the night". One of the poems our model generated was:

> In the night; \
> when the stars shine on her head.\
> the mounds are deep, \
> and the water's dark, \
> and the water's cold \
> and with her hand,\
> with her lips, \
> in song and song,\
> the sound of the birds

This can easily be applied to other datasets. Feel free to train your own adapter and upload it at the [Hub](https://adapterhub.ml/) or browse the adapters trained by the community.

# Conclusion

The new version 2.0 of the AdapterHub framework supports adapters for GPT-2 and BART. The support of these two models offers new possibilities in solving sequence to sequence tasks with adapters. To checkout AdapterHub and its other features visit us on [Github](https://github.com/Adapter-Hub/adapter-transformers).



# References
- Bapna, A., Arivazhagan, N., & Firat, O. (2019). Simple, scalable adaptation for neural machine translation. arXiv preprint arXiv:1909.08478.
- Chen, W., Chen, J., Su, Y., Chen, Z., & Wang, W. Y. (2020). Logical natural language generation from open-domain tables. arXiv preprint arXiv:2004.10404.
- Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. arXiv preprint arXiv:1506.03340.
- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q.D., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. ICML.
- Narayan, S., Cohen, S. B., & Lapata, M. (2018). Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. arXiv preprint arXiv:1808.08745.
- Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I. (2020). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. ArXiv, abs/2005.00247.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. arXiv preprint arXiv:1704.04368.
- Sheng, E., & Uthus, D. (2020). Investigating Societal Biases in a Poetry Composition System. arXiv preprint arXiv:2011.02686.
