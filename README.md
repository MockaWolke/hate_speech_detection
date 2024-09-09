# Hate Speech Detection - Local Inference

This repository contains example code for performing local inference using Hugging Face's transformers models. You can run this on a CPU with or without ONNX acceleration.

## Install ONNX Runtime 

To install the required packages, it is recommended to use an environment manager like Miniconda (https://docs.anaconda.com/miniconda/miniconda-install/). Anaconda is also suitable.

You can start with our environment:

``` conda create --file onnx_env.yml  ```


It might not work on another machine, so here are the steps to create your own

```
conda create -n hsd -y
conda activate hsd 
conda install python==3.10 
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu pip install transformers optimum[exporters,onnxruntime]
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
