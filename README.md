# Hate Speech Detection - Local Inference

This repository contains example code for performing local inference using Hugging Face's transformer models. This can be used for almost every existing model as well as for some Hate Speech Detection ones.

Often, running large language models (LLMs) on a CPU can be quite slow. Fortunately, the ONNX runtime can often provide a 2-3x speedup. There is some example code to run a model with ONNX and also with simple PyTorch.

If you have an NVIDIA GPU or an Apple Silicon Mac, you can achieve even faster inference. Setup tutorials are also provided at the bottom.

## Install ONNX Runtime 


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