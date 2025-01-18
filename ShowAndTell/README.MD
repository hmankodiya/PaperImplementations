# Show and Tell: A Neural Image Caption Generator (Modern Adaptation)

[https://arxiv.org/pdf/1411.4555](https://arxiv.org/pdf/1411.4555)

The **"Show and Tell"** model introduced by Vinyals et al. is a pioneering approach to image captioning, where deep learning techniques are utilized to automatically generate descriptive captions for images. The model employs a convolutional neural network (CNN) for image feature extraction and a recurrent neural network (RNN), specifically a Long Short-Term Memory (LSTM), for sequence generation. This architecture achieves a seamless integration of vision and language, making it one of the first end-to-end trainable systems for image captioning.

In this repository, I have adopted a similar architecture but modernized it by integrating **DINOv2**, a state-of-the-art vision transformer (ViT) model, as the encoder. DINOv2 provides advanced image feature extraction capabilities, capturing fine-grained spatial and semantic details for improved performance in vision tasks. For text processing, I have replaced the traditional RNN decoder with a **GPT-2 tokenizer**, leveraging its robust and efficient handling of text sequences. These updates align the model with modern advancements in both vision and language processing, resulting in a more powerful and adaptable framework for image captioning.

To train and evaluate this model, the **MS COCO 2014 dataset** is used, providing a rich resource of high-quality images and corresponding captions necessary for benchmarking.
The **MS COCO 2014 dataset** is utilized for training and evaluation, as it provides high-quality images and captions essential for benchmarking.

---

### Installation Instructions for MS COCO 2014 Dataset and Annotations

#### 1. **Download MS COCO Dataset Images**
Use the script below to download a subset of the **MS COCO 2014 dataset images**:

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO 2014 validation dataset with captions
train_dataset = foz.load_zoo_dataset(
    "coco-2014",
    split="train",
    max_samples=1000,
)

val_dataset = foz.load_zoo_dataset(
    "coco-2014",
    split="validation",
    max_samples=450,
)

test_dataset = foz.load_zoo_dataset(
    "coco-2014",
    split="test",
    max_samples=450,
)
# Optional: Launch the FiftyOne app to explore the dataset
session = fo.launch_app(dataset)
```

This script uses FiftyOne's `load_zoo_dataset` method to download a sample of 10 test images from the **MS COCO 2014 dataset**. Adjust `max_samples` to download more images as needed.

#### 2. **Download Annotations**
Download the MS COCO 2014 annotations using the following steps:

- **Via Curl:**
   ```bash
   curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
   ```

- **Extract the Annotations:**
   ```bash
   unzip annotations_trainval2014.zip
   ```

This will create a folder containing files such as:
- `captions_train2014.json`
- `captions_val2014.json`
- `instances_train2014.json`
- `instances_val2014.json`

- **Organize the Files:**
   ```bash
   mkdir -p ~/datasets/coco2014
   mv annotations ~/datasets/coco2014/
   ```

- **Verify the Annotations:**
   ```bash
   ls ~/datasets/coco2014/annotations
   ```
---


Classes: airplane, boat