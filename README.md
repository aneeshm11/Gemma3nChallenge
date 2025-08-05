# PROJECT OVERVIEW
This project processes Whole Slide Images (WSI) for pathology tasks and introduces a novel approach for training Vision-Language Models (VLMs) for medical image analysis. It trains a large language model (LLM) (Gemma-3n) through its NLP/language layers to process image tokens in an autoregressive manner (without using it's vision modality), producing concise description / reports for pathology images that capture the cellular details of body organs stored in WSI files. This project demonstrates significantly improved performance compared to CLIP and SigLIP (both of which were also compared under the same training method by processing image tokens in an autoregressive manner). All processes are conducted using Jupyter notebooks on Kaggle, with additional training done on Modal website. The workflow involves downloading raw WSI data, processing it, training various models, and performing inference. This project paves the way for future research into adapting LLMs for visual tasks involving domain-specific data distributions that differ significantly from the datasets used to pretrain these models. 

A detailed technical report is included in this repository, with and in-depth explanations of all components of the project. For a comprehensive understanding, make sure to give it a thorough read.
Demo Video:
```
https://www.youtube.com/watch?v=2rA48SUgUwE
```

# Dataset Files

The datasets are available in the following links, all are publicly available and consist of the RAW WSI files without any downscaled operations. 

## HuggingFace Datasets

```
https://huggingface.co/datasets/aneeshm44/reg1
https://huggingface.co/datasets/aneeshm44/reg2
https://huggingface.co/datasets/aneeshm44/reg3
https://huggingface.co/datasets/aneeshm44/reg4
https://huggingface.co/datasets/aneeshm44/reg5
``` 

## Kaggle Datasets
 
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


# CODE FILES and workflow

All code files are in Jupyter notebook format. The files on GitHub represent the latest versions, while all older versions are available on Kaggle and can be accessed publicly.

The order of usage of all files in this project is listed below:


> 1) Downloading data from FTP server and uploading to HuggingFace
```
https://www.kaggle.com/code/aneeshmukkamala/data-download-to-hf
```

> 2) Downloading dataset from HuggingFace to Kaggle

NOTE:
This intermediate step is necessary due to Kaggle’s limited disk space of 19.5 GB, compared to 100 GB (or 200 GB for TPU sessions) available on Colab. The free tier of Colab lacks the “Save and run all” feature that Kaggle provides, making Kaggle better suited for long-running tasks that run in the background

Specifically, Kaggle is used to handle the initial slow phase of downloading files from an FTP server, a process that can take 10–12 hours. Kaggle’s background execution via commit sessions makes this very simple. Once 19.5 GB of data is downloaded, it is uploaded to Hugging Face as a single commit. The disk is then cleared, and the process continues in a loop until all the data has been transferred.

After the full dataset is uploaded to Hugging Face, it is downloaded to Colab and used to create multiple datasets on Kaggle via the Kaggle API. This entire workflow ensures that no downloads are done on the local machine, everything is handled using Kaggle and Colab for this step

```
https://www.kaggle.com/code/aneeshmukkamala/data-download-hf-to-kaggle
```

> 3) EDA of the WSI dimensions of all files
```
https://www.kaggle.com/code/aneeshmukkamala/edareg
```

> 4) Processing raw WSI into processed images for training
```
https://www.kaggle.com/code/aneeshmukkamala/data-processing
```

> 5) Train Custom ViT and CNN models on processed WSI
```
https://www.kaggle.com/code/aneeshmukkamala/vm-scratch
```

> 6) Train TIMM backbone ViT and CNN models on processed WSI
```
https://www.kaggle.com/code/aneeshmukkamala/vm-timm
```

> 7) Perform end to end training on LLM and projector moduels using Unsloth and Pytorch Lightning
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-train-runs
```

> 8) Code to merge LoRA adapters back into the base LLM
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-merger
``` 

> 9) Perform inference on all modules (the LLM, projector models, image model)
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-inference-1
```

> 10) Get metrics like ROGUE, LEVENSHTEIN ration and similarity scores between ground truth and output reports during inference for comparison
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-metrics/
```

> 11) Training curves are plotted using .npy files that were used to track the step-wise training losses.
```
https://www.kaggle.com/code/aneeshmukkamala/vlm-plots/
```

# These are offline dependcies and are added to system path as L4 GPUs do not offer internet access. 
```
https://www.kaggle.com/code/aneeshmukkamala/pip-vlmfs
https://www.kaggle.com/code/aneeshmukkamala/pip-unsloth
https://www.kaggle.com/code/aneeshmukkamala/pip-openslide
```


# Datasets used for training and inference. 

> Base weights of timm models for offline access into L4 GPUs
```
https://www.kaggle.com/datasets/aneeshmukkamala/codefiles/
```

> Trained Timm model weights
```
https://www.kaggle.com/datasets/aneeshmukkamala/timmweights/
```

NOTE:
The data processing was carried out in mutliple CPU sessions (18 different sessions) on Kaggle. The session wise data (the images in .npy format, the labels) are stored in this dataset as well. This can be found in the sessionfiles dir in this dataset below

> Processed WSI training data
```
https://www.kaggle.com/datasets/aneeshmukkamala/miccaireg/
```

> Gemma model weights downloaded for offline access into L4 GPUs
```
https://www.kaggle.com/datasets/aneeshmukkamala/gemma3/
```

> Training weights having LoRA adapters, projector pth files.
```
https://www.kaggle.com/datasets/aneeshmukkamala/lmweights/
```

> Training logs of all models were collected into .npy files and the graphs were plotted locally. This dataset consists the .npy files, images and the parquet files having responses of the final trained LLMs to get metrics
```
https://www.kaggle.com/datasets/aneeshmukkamala/outputsandlogs
```

> Float 32 trained variant
```
https://www.kaggle.com/datasets/aneeshmukkamala/float32gemma3n
```


# Complete Inference Package

```
https://www.kaggle.com/datasets/aneeshmukkamala/inferencedata/
```



All-in-one inference dataset and code (contains all necessary components from above datasets)

```https://www.kaggle.com/code/aneeshmukkamala/vlm-inference-1``` 

This notebook listed above is used for inference. 
The float32 variant model is also attatched to this notebook. 

# Repository Structure
    
    codefiles/
        train_and_inference/

            Contains notebooks for training, running inference, merging LoRA weights.

        utilityscripts/

            Utility scripts to install packages dependencies required for offline training on L4 GPUs. This can be used on other GPUs as well to skip the package downloading step

    logs_and_metrics/

        Plots and images of visualizations for analyzing training and evaluation metrics.

    modalapp/

        Code for training models using GPUs on Modal (https://modal.com/) 

        new/
            
            Implements training logic for the models based on the newly proposed method.

        old/
   
            Contains older experiments using models like CLIP, SigLIP, and other TIMM-based variants.
    
    utils/
   
        Scripts for data downloading, preprocessing, and some exploratory data analysis.


