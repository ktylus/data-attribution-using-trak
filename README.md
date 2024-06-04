# Data Attribution using TRAK
### Project for Automating Science using Deep Learning course at Jagiellonian University

---

### Project description
We used [TRAK](https://arxiv.org/abs/2303.14186) to identify how important each example from the training set was for the model.
We then tried removing a small fraction of 1. most positively infuential examples, 2. most negatively influential examples (detrimental for training) identified using
a [data debiasing method](https://openreview.net/pdf?id=Agekm5fdW3).
In addition to that, we used raw TRAK scores to see if we could gather any insights looking at the examples with the lowest and highest scores, for a given data point.

The model we used was a pretrained [ResNet-18](https://huggingface.co/microsoft/resnet-18).
We experimented with it by finetuning it on a 10-class subset of the [Food101](https://huggingface.co/datasets/nateraw/food101) dataset, which purposefully has some examples
that are mislabeled. Our hope was that they could be identified using TRAK.

### Results summary
The results can be seen [here](https://docs.google.com/document/d/1epswAMVI0OixPNPi5R153WwKyNFZBPALeetgHCTlZLE).

### Using the code

Required packages are listed in requiremets.txt file. User is advised to use tools to create appriopriate working environmet, e.g. conda or pip:

```
pip install -r requirements.txt
```

The code is meant to be self-explanatory, lengthy implementation details are packed into modules. In code we use [TRAK](https://github.com/MadryLab/trak) repository. In particular, we implement a class inheriting from TRAK's AbstractModelOutput. Please notice that our implementation differs from the default [ImageClassificationModelOutput](https://trak.readthedocs.io/en/latest/modeloutput.html#implementation) very slightly - the only modification is a result of ResNet-18's output type, out of which we must treat output.logits as logits, to preserve TRAK's logic.

As TRAK is parametrized, it is advised to become familiar with TRAK's documentation to get the best out of it in your application/field of interest.

Depending on the hardware and software you possess, you are free to play around with more or less *computationally demanding* settings (including TRAK's settings). Although most of the notebook does not demand GPU to execute as it is, **TRAKer does**, so GPU is essential to reproduce all outcomes we have achieved.

In case of any doubts, feel free to contact us via **issues**.

Best regards,
Authors
