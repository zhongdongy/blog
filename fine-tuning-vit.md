---
title: "Fine-Tune ViT for Image Classification with 🤗 Transformers"
thumbnail: /blog/assets/15_fine_tune_wav2vec2/wav2vec2.png
---

<h1>
    Fine-Tune ViT for Image Classification with 🤗 Transformers
</h1>

<div class="blog-metadata">
    <small>Published August 27, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/fine-tune-vit.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/nateraw">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/32437151?s=400&u=4ec59abc8d21d5feea3dab323d23a5860e6996a4&v=4" title="Gravatar">
        <div class="bfc">
            <code>nateraw</code>
            <span class="fullname">Nate Raw</span>
        </div>
    </a>
</div>

The Vision Transformer (ViT) was introduced in [June 2021](https://arxiv.org/abs/2010.11929) by a team of researchers at Google Brain. 

🚧 **TODO - discuss model more here** 🚧

![vit_figure.png](https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_figure.png)

🚧 **TODO - discuss model more here** 🚧

In this notebook, we'll walk through how to leverage 🤗 `datasets` to download and process image classification datasets, and then use them to fine-tune a pre-trained ViT with 🤗 `transformers`.

---

Before we do anything else, lets install both of those packages.

```
! pip install transformers datasets
```

## Load a Dataset

We'll start by loading a small image classification dataset and taking a look at its structure.

For the sake of this tutorial, we'll use [beans](https://huggingface.co/datasets/beans) - a small dataset containing images of both healthy and unhealthy bean leaves (and their associated labels). 🍃

```python
from datasets import load_dataset

ds = load_dataset('beans')
print(ds)
```

**Print Output:**

```
DatasetDict({
    train: Dataset({
        features: ['image_file_path', 'labels'],
        num_rows: 1034
    })
    validation: Dataset({
        features: ['image_file_path', 'labels'],
        num_rows: 133
    })
    test: Dataset({
        features: ['image_file_path', 'labels'],
        num_rows: 128
    })
})
```

Image datasets within the datasets package often have a feature representing the path to individual image files. Let's look at an example:

```python
ex = ds['train'][0]
ex
```

**Print Output:**

```
{
    'image_file_path': '/root/.cache/huggingface/datasets/downloads/extracted/0aaa78294d4bf5114f58547e48d91b7826649919505379a167decb629aa92b0a/train/angular_leaf_spot/angular_leaf_spot_train.304.jpg',
    'labels': 0
}
 ```

As we can see, image_file_path is a path within a cache directory that was created when we downloaded the source dataset.

Lets take that path and open the file as a `PIL` image so we can take a look at it. 👀

```python
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        image = Image.open(f).convert("RGB")
    return image

image = pil_loader(ex['image_file_path'])
image
```

**Print Output:**

![angular_leaf_spot_train.304.jpg](/blog/assets/25_vit/angular_leaf_spot_train.304.jpg)


Thats definitely a leaf! But what kind? 😅

Since the `'labels'` feature of this dataset is a [`datasets.features.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.ClassLabel), we can use it to lookup the corresponding name for this example's label ID.

First, lets access the feature definition for the `'labels'`

```python
labels = ds['train'].features['labels']
labels
```

**Print Output:**

```
ClassLabel(num_classes=3, names=['angular_leaf_spot', 'bean_rust', 'healthy'], names_file=None, id=None)
```

Now, lets print out the class label for our example by using [`ClassLabel.int2str`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.ClassLabel.int2str).

```python
labels.int2str(ex['labels'])
```

**Print Output:**

```
'angular_leaf_spot'
```

Turns out the leaf shown above is infected with Angular Leaf Spot, a serious disease in bean plants. 😢

To get to know this dataset a little better, we can write a function that'll display a grid of examples from each class so we can get a better idea of what we're working with.

```python
from PIL import ImageDraw, ImageFont

def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds['train'].features['labels'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    for label_id, label in enumerate(labels):

        # Filter by label, shuffle, and grab a few samples
        ds_slice = ds['train'] \
            .filter(lambda ex: ex['labels'] == label_id) \
            .shuffle(seed) \
            .select(range(examples_per_class))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = pil_loader(example['image_file_path'])
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)

    return grid

show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)
```

**Print Output:**

![bean_image_grid.png](/blog/assets/25_vit/bean_image_grid.png)

You can run the above code a few times to see other random examples from the dataset.

We can see that:

- Angular Leaf Spot: Has irregular brown patches
- Bean Rust: Has circular brown spots surrounded with a white-ish yellow ring
- Healthy: ...looks healthy 😅

## Loading ViT Feature Extractor

Now that we know what our images look like and have a better understanding of the problem we're trying to solve, let's see how we can prepare these images for our model.

When ViT models are trained, specific transformations are applied to images being fed into them. 

⚠️ If you use the wrong transformations, the model won't be able to understand what it's seeing! ⚠️

To make sure we apply the correct transformations, we'll use a `ViTFeatureExtractor` initialized from a configuration that was saved alongside the pretrained model we plan to fine-tune. In our case, we'll be using the [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) model, so lets load its corresponding feature extractor from the 🤗 Hub.

```python
from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
```

If we print a feature extractor, we can see its configuration:

```
ViTFeatureExtractor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

To process an image, simply pass it to the feature extractor's call function. This will return pixel values, which is the numeric representation of your image that we'll pass to the model.

We get a numpy array by default, but if we add the `return_tensors='pt'` argument, we'll get back torch tensors instead.

```python
feature_extractor(image, return_tensors='pt')
```

**Print Output:**

```
{'pixel_values': tensor([[[[ 0.3647,  0.3647,  0.3804,  ..., -0.7647, -0.7569, -0.7490],
          [ 0.3804,  0.3961,  0.4118,  ..., -0.6157, -0.6706, -0.7176],
          [ 0.2627,  0.3098,  0.3176,  ..., -0.4275, -0.5216, -0.6392],
          ...,
          [-0.4588, -0.4588, -0.5059,  ..., -0.5373, -0.4588, -0.3647],
          [-0.4980, -0.4824, -0.5216,  ..., -0.3490, -0.1922, -0.1765],
          [-0.4980, -0.5137, -0.4824,  ..., -0.4039, -0.3412, -0.4275]],

         [[ 0.7098,  0.7176,  0.7333,  ..., -0.8745, -0.8196, -0.7961],
          [ 0.7412,  0.7569,  0.7725,  ..., -0.7333, -0.7647, -0.8039],
          [ 0.6549,  0.6863,  0.6863,  ..., -0.5922, -0.6627, -0.7647],
          ...,
          [ 0.0824,  0.0745,  0.0118,  ..., -0.7020, -0.6941, -0.6235],
          [ 0.0196,  0.0039, -0.0275,  ..., -0.5294, -0.4588, -0.4667],
          [-0.0275, -0.0902, -0.0588,  ..., -0.5686, -0.5373, -0.6157]],

         [[ 0.4667,  0.4824,  0.5059,  ..., -0.9922, -0.9922, -0.9843],
          [ 0.4353,  0.4902,  0.5373,  ..., -0.9137, -0.9529, -0.9843],
          [ 0.3020,  0.3804,  0.4353,  ..., -0.8118, -0.8510, -0.9373],
          ...,
          [-0.8824, -0.8745, -0.9294,  ..., -0.9216, -0.8902, -0.8196],
          [-0.9137, -0.9137, -0.9529,  ..., -0.7333, -0.6549, -0.6471],
          [-0.9373, -0.9686, -0.9451,  ..., -0.7647, -0.7333, -0.8118]]]])}
```

## Processing the Dataset

Now that we know how to read in images and transform them into inputs, let's write a function that will put those two things together to process a single example from the dataset. 💪

```python
def process_example(example):
    image = pil_loader(example['image_file_path'])
    example.update(feature_extractor(image), return_tensors='pt')
    return example

process_example(ds['train'][0])
```

**Print Output:**


```
{'image_file_path': '/root/.cache/huggingface/datasets/downloads/extracted/0aaa78294d4bf5114f58547e48d91b7826649919505379a167decb629aa92b0a/train/healthy/healthy_train.25.jpg',
 'labels': 2,
 'pixel_values': [array([[[ 0.36470592,  0.36470592,  0.3803922 , ..., -0.7647059 ,
           -0.75686276, -0.7490196 ],
          [ 0.3803922 ,  0.39607847,  0.41176474, ..., -0.6156863 ,
           -0.67058825, -0.7176471 ],
          [ 0.26274514,  0.30980396,  0.3176471 , ..., -0.42745095,
           -0.52156866, -0.6392157 ],
          ...,
          [-0.4588235 , -0.4588235 , -0.5058824 , ..., -0.5372549 ,
           -0.4588235 , -0.36470586],
          [-0.4980392 , -0.4823529 , -0.52156866, ..., -0.3490196 ,
           -0.19215685, -0.17647058],
          [-0.4980392 , -0.5137255 , -0.4823529 , ..., -0.40392154,
           -0.34117645, -0.42745095]],
  
         [[ 0.70980394,  0.7176471 ,  0.73333335, ..., -0.8745098 ,
           -0.81960785, -0.79607844],
          [ 0.7411765 ,  0.75686276,  0.77254903, ..., -0.73333335,
           -0.7647059 , -0.8039216 ],
          [ 0.654902  ,  0.6862745 ,  0.6862745 , ..., -0.5921569 ,
           -0.6627451 , -0.7647059 ],
          ...,
          [ 0.082353  ,  0.07450986,  0.01176476, ..., -0.7019608 ,
           -0.69411767, -0.62352943],
          [ 0.0196079 ,  0.00392163, -0.02745098, ..., -0.5294118 ,
           -0.4588235 , -0.46666664],
          [-0.02745098, -0.09019607, -0.05882353, ..., -0.5686275 ,
           -0.5372549 , -0.6156863 ]],
  
         [[ 0.4666667 ,  0.48235297,  0.5058824 , ..., -0.99215686,
           -0.99215686, -0.9843137 ],
          [ 0.43529415,  0.4901961 ,  0.5372549 , ..., -0.9137255 ,
           -0.9529412 , -0.9843137 ],
          [ 0.30196083,  0.3803922 ,  0.43529415, ..., -0.8117647 ,
           -0.8509804 , -0.9372549 ],
          ...,
          [-0.88235295, -0.8745098 , -0.92941177, ..., -0.92156863,
           -0.8901961 , -0.81960785],
          [-0.9137255 , -0.9137255 , -0.9529412 , ..., -0.73333335,
           -0.654902  , -0.64705884],
          [-0.9372549 , -0.96862745, -0.94509804, ..., -0.7647059 ,
           -0.73333335, -0.8117647 ]]], dtype=float32)],
 'return_tensors': 'pt'}
 ```

 While we could call [`ds.map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) and apply this to every example at once, this can be very slow. Instead, we'll apply a ***transform*** to the dataset. Transforms are only applied to examples as you index them.

First, though, we'll need to update our last function to accept a ***batch*** of data, as that's what [`ds.with_transform`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) expects.

```python
def transform(example_batch):

    # Read in all the image file paths as PIL images
    images = []
    for path in example_batch.pop('image_file_path'):
        image = pil_loader(path)
        images.append(image)

    # Add pixel values to batch by passing images through the feature extractor
    example_batch.update(feature_extractor(images, return_tensors='pt'))

    return example_batch
```

We can directly apply this to our dataset using `ds.with_transform(transform)`.


```python
prepared_ds = ds.with_transform(transform)
```

Now, whenever we get an example from the dataset, our transform will be 
applied in real time (on both samples and slices, as shown below)

```python
prepared_ds['train'][0:2]
```

**Print Output:**

```
{'labels': [2, 2],
 'pixel_values': tensor([[[[ 0.3647,  0.3647,  0.3804,  ..., -0.7647, -0.7569, -0.7490],
           [ 0.3804,  0.3961,  0.4118,  ..., -0.6157, -0.6706, -0.7176],
           [ 0.2627,  0.3098,  0.3176,  ..., -0.4275, -0.5216, -0.6392],
           ...,
           [-0.4588, -0.4588, -0.5059,  ..., -0.5373, -0.4588, -0.3647],
           [-0.4980, -0.4824, -0.5216,  ..., -0.3490, -0.1922, -0.1765],
           [-0.4980, -0.5137, -0.4824,  ..., -0.4039, -0.3412, -0.4275]],
 
          [[ 0.7098,  0.7176,  0.7333,  ..., -0.8745, -0.8196, -0.7961],
           [ 0.7412,  0.7569,  0.7725,  ..., -0.7333, -0.7647, -0.8039],
           [ 0.6549,  0.6863,  0.6863,  ..., -0.5922, -0.6627, -0.7647],
           ...,
           [ 0.0824,  0.0745,  0.0118,  ..., -0.7020, -0.6941, -0.6235],
           [ 0.0196,  0.0039, -0.0275,  ..., -0.5294, -0.4588, -0.4667],
           [-0.0275, -0.0902, -0.0588,  ..., -0.5686, -0.5373, -0.6157]],
 
          [[ 0.4667,  0.4824,  0.5059,  ..., -0.9922, -0.9922, -0.9843],
           [ 0.4353,  0.4902,  0.5373,  ..., -0.9137, -0.9529, -0.9843],
           [ 0.3020,  0.3804,  0.4353,  ..., -0.8118, -0.8510, -0.9373],
           ...,
           [-0.8824, -0.8745, -0.9294,  ..., -0.9216, -0.8902, -0.8196],
           [-0.9137, -0.9137, -0.9529,  ..., -0.7333, -0.6549, -0.6471],
           [-0.9373, -0.9686, -0.9451,  ..., -0.7647, -0.7333, -0.8118]]],
 
 
         [[[ 0.1922,  0.1686,  0.1137,  ...,  0.1137,  0.1373,  0.1451],
           [ 0.2392,  0.2000,  0.1451,  ...,  0.1059,  0.1294,  0.1294],
           [ 0.3804,  0.3098,  0.2314,  ...,  0.0902,  0.1137,  0.1059],
           ...,
           [ 0.3961,  0.3961,  0.4196,  ..., -0.0196, -0.0510, -0.1686],
           [ 0.3882,  0.4275,  0.4118,  ..., -0.0353, -0.0275, -0.2078],
           [ 0.3961,  0.4431,  0.4431,  ..., -0.0745,  0.0118, -0.1843]],
 
          [[ 0.4118,  0.4118,  0.3804,  ...,  0.3725,  0.3882,  0.3882],
           [ 0.4510,  0.4275,  0.3882,  ...,  0.3725,  0.3647,  0.3647],
           [ 0.5294,  0.4824,  0.4275,  ...,  0.3569,  0.3490,  0.3255],
           ...,
           [-0.2078, -0.2157, -0.1922,  ...,  0.1608,  0.1216,  0.0196],
           [-0.2392, -0.2000, -0.2078,  ...,  0.1451,  0.1373, -0.0196],
           [-0.2392, -0.1922, -0.1843,  ...,  0.1059,  0.1765,  0.0039]],
 
          [[-0.0353, -0.0745, -0.1373,  ..., -0.0745, -0.0745, -0.0745],
           [-0.0039, -0.0588, -0.1216,  ..., -0.0902, -0.0824, -0.0824],
           [ 0.0980,  0.0275, -0.0510,  ..., -0.1137, -0.0902, -0.1059],
           ...,
           [ 0.1059,  0.1059,  0.1373,  ..., -0.0353, -0.0510, -0.1529],
           [ 0.0824,  0.1216,  0.1137,  ..., -0.0588, -0.0275, -0.1843],
           [ 0.0824,  0.1294,  0.1373,  ..., -0.0902,  0.0039, -0.1686]]]])}
```

# Training and Evaluation

The data is processed and we are ready to start setting up the training pipeline. We will make use of 🤗's Trainer, but that'll require us to do a few things first:

- Define a collate function. By default, our batches will be put together as lists of individual example dicts. We need to collate them into individual batch tensors for both `pixel_values` and `labels`.

- Define an evaluation metric. During training, the model should be evaluated on its prediction accuracy. We should define a compute_metrics function accordingly.

- Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.

- Define the training configuration.

When we finish fine-tuning the model, we'll evaluate it on the `validation` data to verify that it has indeed learned to correctly classify our images.

### Define our data collator

Batches are coming in as lists of dicts, so we just unpack and stack those into batch tensors.

We return a batch `dict` from our `collate_fn` so we can simply `**unpack` the inputs to our model later. ✨

```python
import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
```

### Define an evaluation metric

Here, we load the accuracy metric from `datasets`. Then, we write a function that takes in a model prediction and computes the accuracy.

```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
```

Now we can finally load our pretrained model!

 - We'll add `num_labels` when initializing it to make sure the model creates a classification head with the right number of output units.
 - We'll also include the `id2label` and `label2id` mappings so we have human readable labels when we share the model on 🤗 hub later.

```python
from transformers import ViTForImageClassification

labels = ds['train'].features['labels'].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
```

### Define training arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./vit-base-beans-demo",
  group_by_length=True,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=2,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)
```

Now we can initialize the trainer given the model, feature extractor, configured training arguments, collate function, and the dataset splits.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)
```

## Training

```python
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```

**Print Output:**

```
ADD THIS
```


## Evaluation

```python
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```

**Print Output:**

```
ADD THIS
```

The resulting model has been shared to [nateraw/vit-base-beans](https://huggingface.co/nateraw/vit-base-beans).
