# Experimenting with TRAK
### Project for Automating Science Using Deep Learning course at Jagiellonian University

---

### Project description
We used [TRAK](https://arxiv.org/abs/2303.14186) to identify how important each example from the training set was for the model.
We then tried removing a small fraction of 1. most positively infuential examples, 2. most negatively influential examples (detrimental for training) identified using
a [data debiasing method](https://openreview.net/pdf?id=Agekm5fdW3).
In addition to that, we used raw TRAK scores to see if we could gather any insights looking at the examples with the lowest and highest scores, for a given data point.

The model we used was a pretrained [ResNet-18.](https://huggingface.co/microsoft/resnet-18)
We experimented with it by finetuning it on a 10-class subset of the [Food101](https://huggingface.co/datasets/nateraw/food101) dataset, which purposefully has some examples
that are mislabeled. Our hope was that they could be identified using TRAK.

### Results summary
The results can be seen [here.](https://docs.google.com/document/d/1epswAMVI0OixPNPi5R153WwKyNFZBPALeetgHCTlZLE)

### Using the code
requirements.txt, instructions, trak github link perhaps

