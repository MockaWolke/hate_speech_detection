# Hate Speech Detection - Local Inference

This repository contains example code for performing local inference using Hugging Face's transformers models. This can be used for almost every existing model as also for some Hate Speech Detection ones.

Often running LLMs on a CPU can be quite slow, luckily there is the ONNX runtime which can often provide a 2-3x speedup. There is some examle code to run a model with onnx and also with simple pytorch.

If you have a Nvidia GPU or an Apple Silicon mac, you can have even faster inference. Some setup tutorial is also at the bottom.

## Install ONNX Runtime 

Verify your install:

If you have linux you can start with our environment:

``` conda create --file onnx_env.yml  ```


It might not work on another operating system or machine. If so you can follow these steps manually.

```
conda create -n hsd -y
conda activate hsd 
conda install python==3.10 

On linux:
pip install -U torch torchvision --index-url https://

Other:
pip install -U torch torchvision --index-url https://


download.pytorch.org/whl/cpu pip install transformers optimum[exporters,onnxruntime]
```

## Exporting a Model to ONNX

First, select a model from the Hugging Face model hub, such as: `Hate-speech-CNERG/bert-base-uncased-hatexplain`.

Then, export it to ONNX using the following command:


```optimum-cli export onnx --model Hate-speech-CNERG/bert-base-uncased-hatexplain --task text-classification your_onnx_model_name```

The following output confirms successful export:

```
....
Validating ONNX model your_onnx_model_name/model.onnx...
	-[✓] ONNX model output names match reference model (logits)
	- Validating ONNX Model output "logits":
		-[✓] (2, 2) matches (2, 2)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: your_onnx_model_name
```

## Inference Example

For an inference example using both ONNX and Torch runtimes, refer to [this notebook](example.ipynb).


## Using CUDA (for NVIDIA) Acceleration

Install torch for CUDA by following this guide: https://pytorch.org/get-started/locally/

And install transfomers:
```pip install transfomers```

It is highly recommended to install within a Conda environment to avoid conflicts with NVIDIA drivers.

Verify your install:

```python
import torch
torch.cuda.is_available()
```
If the output is False, something went wrong.

For running models on CUDA:

```
from transformers.pipelines import pipeline as torch_pipeline

model_name = "Hate-speech-CNERG/dehatebert-mono-english"

torch_pipe = torch_pipeline("text-classification", model_name, device=0)  # Adjust the device index based on your NVIDIA GPU

result = torch_pipe("This is very nice")
```


## Using Apple MPS accelartion

Install torch for mps giving [this guide.](https://pytorch.org/get-started/locally/)

And install transfomers:
```pip install transfomers```



Verify your install:
```
import torch
torch.backends.mps.is_available()
```
If the output is False, something went wrong.

- Output should be True (if False something went wrong)


```
from transformers.pipelines import pipeline as torch_pipeline

model_name = "Hate-speech-CNERG/dehatebert-mono-english"

torch_pipe = torch_pipeline("text-classification", model_name, device = "mps") 

torch_pipe("This is very nice")
```

Note: Documentation for mps was a bit sparse and we could not test it.