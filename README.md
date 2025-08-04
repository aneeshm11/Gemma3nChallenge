# DATASET FILES

The datasets are available in the following links, all are publicly available and consist of the RAW WSI files without any downscaled operations. 

## HUGGINGFACE

```
https://huggingface.co/datasets/aneeshm44/reg1
https://huggingface.co/datasets/aneeshm44/reg2
https://huggingface.co/datasets/aneeshm44/reg3
https://huggingface.co/datasets/aneeshm44/reg4
https://huggingface.co/datasets/aneeshm44/reg5
``` 

## KAGGLE
 
```
https://www.kaggle.com/datasets/aneeshmukkamala/reghf1
https://www.kaggle.com/datasets/aneeshmukkamala/reghf2
https://www.kaggle.com/datasets/aneeshmukkamala/reghf3
https://www.kaggle.com/datasets/aneeshmukkamala/reghf4
https://www.kaggle.com/datasets/aneeshmukkamala/reghf6
https://www.kaggle.com/datasets/aneeshmukkamala/reghf7
https://www.kaggle.com/datasets/aneeshmukkamala/reghf8
https://www.kaggle.com/datasets/aneeshmukkamala/reghf9
https://www.kaggle.com/datasets/aneeshmukkamala/reghf10
https://www.kaggle.com/datasets/aneeshmukkamala/reghf11
```


# CODE FILES

All code files are in jupyter notebook format only as all processes for this project are carried out on Kaggle. 
Except the files used for training an additional model on Modal website 

EDA of the WSI dimensions
```
https://www.kaggle.com/code/aneeshmukkamala/edareg
```

Code to merge LoRA adapters back into the model
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-merger
``` 

Perform inference on all modules (the LLM, projector models, image model)
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-inference-1
``` 

Perform end to end training using Unsloth and Pytorch Lightning
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-train-runs
``` 

Train Custom ViT and CNN models on downscaled WSI
```
https://www.kaggle.com/code/aneeshmukkamala/vm-scratch
```

Train TIMM backbone ViT and CNN models on downscaled WSI
```
https://www.kaggle.com/code/aneeshmukkamala/vm-timm
```


# These are offline dependcies and are added to system path as L4 GPUs do not offer internet access. 
```
https://www.kaggle.com/code/aneeshmukkamala/pip-vlmfs
https://www.kaggle.com/code/aneeshmukkamala/pip-unsloth
https://www.kaggle.com/code/aneeshmukkamala/pip-openslide
```



# Datasets used for training and inference. 

Weights of timm models for offline access into L4 GPUs
```
https://www.kaggle.com/datasets/aneeshmukkamala/codefiles/
```

Timm model weights
```
https://www.kaggle.com/datasets/aneeshmukkamala/timmweight/
```

Downscaled WSI training data
```
https://www.kaggle.com/datasets/aneeshmukkamala/miccaireg/
```

Model weights downloaded offline access into L4 GPUs
```
https://www.kaggle.com/datasets/aneeshmukkamala/gemma3/
```

Training weights having LoRA adapters, projector pth files.
```
https://www.kaggle.com/datasets/aneeshmukkamala/lmweights/
```

The above listed Kaggle datasets have weights from various runs, different combination and many more.
All final needed data to run inference is bundled in this single dataset below using the above datasets,

```
https://www.kaggle.com/datasets/aneeshmukkamala/inferencedata/
```
