# NLU Model: Voice Intent Recognition

This repository contains the Natural Language Understanding (NLU) pipeline for a private voice assistant. It transforms raw text (from STT) into actionable intents using a lightweight, optimized Deep Learning approach.

## Overview
The model is designed to run **100% offline** on Android devices. It uses a **Bi-Directional LSTM** architecture to capture context from voice commands, supporting intents like calling, messaging, and app navigation.

### Key Features
- **Architecture**: Bi-LSTM (Bidirectional Long Short-Term Memory).
- **Format**: TensorFlow Lite (.tflite) with **Flex Ops** support.
- **Optimization**: Post-training quantization for mobile efficiency.
- **Language**: English/French support (depending on the dataset used).

## Repository Structure
- `src/`: Python scripts for training, evaluation, and TFLite conversion.
- `models/`: Saved model files (`.h5` and `.tflite`).
- `data/`: Training datasets and preprocessing assets.
- `exports/`: `vocab.json` and `intent_map.json` required for mobile integration.

## Technical Stack
- **Python 3.10+**
- **TensorFlow / Keras**: Model training and conversion.
- **NumPy & JSON**: Data preprocessing.