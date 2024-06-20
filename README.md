# Grammar-Synthesis-Enhanced: FLAN-t5

<a href="https://colab.research.google.com/gist/pszemraj/5dc89199a631a9c6cfd7e386011452a0/demo-flan-t5-large-grammar-synthesis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This model is a fine-tuned version of [pszemraj/flan-t5-large-grammar-synthesis](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis) using the C4 200M dataset for the NaraSpeak Bangkit 2024 ENTR-H130 application.

## T5 Model Overview

The T5 (Text-To-Text Transfer Transformer) model, introduced by Google Research, is a transformer-based model that treats every NLP task as a text-to-text problem. This unified approach allows T5 to excel at a variety of tasks, such as translation, summarization, and question answering, by converting inputs and outputs into text format.

### Transformer Architecture

Transformers are a type of deep learning model designed for sequence-to-sequence tasks. They utilize a mechanism called "attention" to weigh the influence of different words in a sequence, allowing the model to focus on relevant parts of the input when generating each word in the output. This architecture is highly parallelizable and has proven effective in NLP tasks.

## Usage in Python

After `pip install transformers`, run the following code:

```python
from transformers import pipeline

corrector = pipeline(
              'text2text-generation',
              'farelzii/GEC_Test_v1',
              )
raw_text = 'i can has cheezburger'
results = corrector(raw_text)
print(results)
