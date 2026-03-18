# BatDetect2 Architecture Overview

This document provides a comprehensive map of the `batdetect2` codebase architecture. It is intended to serve as a deep-dive reference for developers, agents, and contributors navigating the project.

`batdetect2` is designed as a modular deep learning pipeline for detecting and classifying bat echolocation calls in high-frequency audio recordings. It heavily utilizes **PyTorch**, **PyTorch Lightning** for training, and the **Soundevent** library for standardized audio and geometry data classes.

The repository follows a configuration-driven design pattern, heavily utilizing `pydantic`/`omegaconf` (via `BaseConfig`) and the Factory/Registry patterns for dependency injection and modularity. The entire pipeline can be orchestrated via the high-level API `BatDetect2API` (`src/batdetect2/api_v2.py`).

---

## 1. Data Flow Pipeline

The standard lifecycle of a prediction request follows these sequential stages, each handled by an isolated, replaceable module:

1. **Audio Loading (`batdetect2.audio`)**: Read raw `.wav` files into standard NumPy arrays or `soundevent.data.Clip` objects. Handles resampling.
2. **Preprocessing (`batdetect2.preprocess`)**: Converts raw 1D waveforms into 2D Spectrogram tensors.
3. **Forward Pass (`batdetect2.models`)**: A PyTorch neural network processes the spectrogram and outputs dense prediction tensors (e.g., detection heatmaps, bounding box sizes, class probabilities).
4. **Postprocessing (`batdetect2.postprocess`)**: Decodes the raw output tensors back into explicit geometry bounding boxes and runs Non-Maximum Suppression (NMS) to filter redundant predictions.
5. **Formatting (`batdetect2.data`)**: Transforms the predictions into standard formats (`.csv`, `.json`, `.parquet`) using `OutputFormatterProtocol`.

---

## 2. Core Modules Breakdown

### 2.1 Audio and Preprocessing
- **`audio/`**: 
  - Centralizes audio I/O using `AudioLoader`. It abstracts over the `soundevent` library, efficiently handling full `Recording` files or smaller `Clip` segments, standardizing the sample rate.
- **`preprocess/`**: 
  - Dictated by the `PreprocessorProtocol`. 
  - Its primary responsibility is spectrogram generation via Short-Time Fourier Transform (STFT).
  - During training, it incorporates data augmentation layers (e.g., amplitude scaling, time masking, frequency masking, spectral mean subtraction) configured via `PreprocessingConfig`.

### 2.2 Deep Learning Models (`models/`)
The `models` directory contains all PyTorch neural network architectures. The default architecture is an Encoder-Decoder (U-Net style) network.
- **`blocks.py`**: Reusable neural network blocks, including standard Convolutions (`ConvBlock`) and specialized layers like `FreqCoordConvDownBlock`/`FreqCoordConvUpBlock` which append normalized spatial frequency coordinates to explicitly grant convolutional filters frequency-awareness.
- **`encoder.py`**: The downsampling path (feature extraction). Builds a sequential list of blocks and captures skip connections.
- **`bottleneck.py`**: The deepest, lowest-resolution segment connecting the Encoder and Decoder. Features an optional `SelfAttention` mechanism to weigh global temporal contexts.
- **`decoder.py`**: The upsampling path (reconstruction), actively integrating skip connections (residuals) from the Encoder.
- **`heads.py`**: Attach to the backbone's feature map to output specific predictions:
  - `BBoxHead`: Predicts bounding box sizes.
  - `ClassifierHead`: Predicts species classes.
  - `DetectorHead`: Predicts detection probability heatmaps.
- **`backbones.py` & `detectors.py`**: Assemble the encoder, bottleneck, decoder, and heads into a cohesive `Detector` model.
- **`__init__.py:Model`**: The overarching wrapper `torch.nn.Module` containing the `detector`, `preprocessor`, `postprocessor`, and `targets`.

### 2.3 Targets and Regions of Interest (`targets/`)
Crucial for training, this module translates physical annotations (Regions of Interest / ROIs) into training targets (tensors).
- **`rois.py`**: Implements `ROITargetMapper`. Maps a geometric bounding box into a 2D reference `Position` (time, freq) and a `Size` array. Includes strategies like:
  - `AnchorBBoxMapper`: Maps based on a fixed bounding box corner/center.
  - `PeakEnergyBBoxMapper`: Identifies the physical coordinate of peak acoustic energy inside the bounding box and calculates offsets to the box edges.
- **`targets.py`**: Reconstructs complete multi-channel target heatmaps and coordinate tensors from the ROIs to compute losses during training.

### 2.4 Postprocessing (`postprocess/`)
- Implements `PostprocessorProtocol`.
- Reverses the logic from `targets`. It scans the model's output detection heatmaps for peaks, extracts the predicted sizes and class probabilities at those peaks, and decodes them back into physical `soundevent.data.Geometry` (Bounding Boxes).
- Automatically applies Non-Maximum Suppression (NMS) configured via `PostprocessConfig` to remove highly overlapping predictions.

### 2.5 Data Management (`data/`)
- **`annotations/`**: Utilities to load dataset annotations supporting multiple standardized schemas (`AOEF`, `BatDetect2` formats).
- **`datasets.py`**: Aggregates recordings and annotations into memory.
- **`predictions/`**: Handles the exporting of model results via `OutputFormatterProtocol`. Includes formatters for `RawOutput`, `.parquet`, `.json`, etc.

### 2.6 Evaluation (`evaluate/`)
- Computes scientific metrics using `EvaluatorProtocol`.
- Provides specific testing environments for tasks like `Clip Classification`, `Clip Detection`, and `Top Class` predictions.
- Generates precision-recall curves and scatter plots.

### 2.7 Training (`train/`)
- Implements the distributed PyTorch training loop via PyTorch Lightning.
- **`lightning.py`**: Contains `TrainingModule`, the `LightningModule` that orchestrates the optimizer, learning rate scheduler, forward passes, and backpropagation using the generated `targets`.

---

## 3. Interfaces and Tooling

### 3.1 APIs
- **`api_v2.py` (`BatDetect2API`)**: The modern API object. It is deeply integrated with dependency injection using `BatDetect2Config`. It instantiates the loader, targets, preprocessor, postprocessor, and model, exposing easy-to-use methods like `process_file`, `evaluate`, and `train`.
- **`api.py`**: The legacy API. Kept for backwards compatibility. Uses hardcoded default instances rather than configuration objects.

### 3.2 Command Line Interface (`cli/`)
- Implements terminal commands utilizing `click`. Commands include `batdetect2 detect`, `evaluate`, and `train`.

### 3.3 Core and Configuration (`core/`, `config.py`)
- **`core/registries.py`**: A string-based Registry pattern (e.g., `block_registry`, `roi_mapper_registry`) that allows developers to dynamically swap components (like a custom neural network block) via configuration files without modifying python code.
- **`config.py`**: Aggregates all modular `BaseConfig` objects (`AudioConfig`, `PreprocessingConfig`, `BackboneConfig`) into the monolithic `BatDetect2Config`.

---

## Summary
To navigate this codebase effectively:
1. Follow **`api_v2.py`** to see how high-level operations invoke individual components.
2. Rely heavily on the typed **Protocols** located in each subsystem's `types.py` module (for example `src/batdetect2/preprocess/types.py` and `src/batdetect2/postprocess/types.py`) to understand inputs and outputs without needing to read each implementation.
3. Understand that data flows structurally as `soundevent` primitives externally, and as pure `torch.Tensor` internally through the network.
