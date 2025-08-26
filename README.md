# MINT-CIKM25: From Intents to Conversations

This repository contains a source code implementation for the paper "From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification" accepted at CIKM 2025.

## Important Note

The original implementation code cannot be shared due to copyright and proprietary restrictions. This repository provides a clean implementation generated based on the methodologies and algorithms described in the published paper.

## Dataset Availability

The MINT-E dataset will be made available at: https://huggingface.co/datasets/fubincom/MINT_E/tree/main (currently under construction)

## Overview

The implementation includes:

1. **Chain-of-Intent**: A novel mechanism combining Hidden Markov Models (HMMs) with Large Language Models (LLMs) for intent-driven dialogue generation
2. **MINT-CL**: A multi-task contrastive learning framework for multi-turn intent classification
3. **Data Processing Utilities**: Comprehensive tools for handling multilingual e-commerce conversation data

## Key Components

### Chain-of-Intent (`chain_of_intent.py`)
- Domain knowledge extraction from historical chat logs
- HMM-based intent sequence sampling
- LLM-enhanced dialogue generation
- Alternative response generation for contrastive learning

### MINT-CL Framework (`mint_cl.py`)
- Hierarchical text classification with label attention
- Multi-task learning with contrastive loss
- Support for multilingual intent classification
- XLM-RoBERTa-based encoder

### Data Processing (`data_utils.py`)
- Conversation data loading and validation
- Train/validation/test splitting
- Multilingual dataset handling
- Intent hierarchy construction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MINT-CIKM25-CRcopy/code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key (for Chain-of-Intent):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Generate Dialogues

Generate intent-driven conversations using Chain-of-Intent:

```bash
python generate_dialogues.py --config config.yaml --num-conversations 1000
```

This will:
- Extract domain knowledge from seed conversations
- Generate new conversations using the Chain-of-Intent mechanism
- Create alternative responses for contrastive learning
- Save generated data for training

### 2. Train MINT-CL Model

Train the multi-task contrastive learning model:

```bash
python train.py --config config.yaml
```

This will:
- Load and preprocess conversation data
- Build intent hierarchy from the data
- Train the MINT-CL model with multi-task contrastive learning
- Evaluate on validation and test sets
- Save model checkpoints and training metrics

### 3. Configuration

Modify `config.yaml` to customize:
- Model parameters (learning rate, batch size, etc.)
- Data paths and preprocessing options
- Chain-of-Intent generation settings
- Evaluation metrics and output formats

## Data Format

### Input Conversation Format

The system expects conversation data in the following format:

```json
{
  "conversation_id": "conv_1",
  "turns": 2,
  "language": "en",
  "domain": "ecommerce",
  "dialogue": [
    {
      "turn": 1,
      "question": "Where is my order?",
      "intent": "track_order",
      "answer": "Let me check your order status."
    },
    {
      "turn": 2,
      "question": "When will it arrive?",
      "intent": "delivery_time", 
      "answer": "It should arrive tomorrow."
    }
  ]
}
```

### Intent Hierarchy

The system supports 3-level hierarchical intent classification:
- **Level 1**: Broad categories (e.g., "order", "product", "payment")
- **Level 2**: Sub-categories (e.g., "order_tracking", "product_info")
- **Level 3**: Specific intents (e.g., "track_order", "delivery_time")

## Architecture Details

### Chain-of-Intent Pipeline

1. **Domain Knowledge Extraction**:
   - Turn distribution P(T)
   - Initial intent distribution P(I₁)
   - Intent transition matrix P(I_t|I_{t-1})

2. **Intent Sequence Sampling**:
   - Sample conversation length from P(T)
   - Sample intent sequence using HMM
   - Generate contextual questions and answers using LLMs

3. **Quality Enhancement**:
   - Generate alternative responses
   - Evaluate response quality
   - Create contrastive pairs for training

### MINT-CL Architecture

1. **Text Encoder**: XLM-RoBERTa-base for multilingual support
2. **Hierarchical Classifier**: 
   - Label attention mechanism
   - Separate heads for each hierarchy level
   - Global hierarchical information integration
3. **Contrastive Learning**:
   - Response ranking task
   - Multi-task loss function
   - Improved representation learning

## Evaluation

The implementation includes comprehensive evaluation:

- **Classification Metrics**: Accuracy, F1-score, Precision, Recall
- **Hierarchical Evaluation**: Performance at each hierarchy level
- **Multilingual Evaluation**: Language-specific performance analysis
- **Cross-lingual Transfer**: Zero-shot evaluation across languages

## File Structure

```
code/
├── chain_of_intent.py     # Chain-of-Intent implementation
├── mint_cl.py            # MINT-CL model and training
├── data_utils.py         # Data processing utilities
├── train.py              # Training script
├── generate_dialogues.py # Dialogue generation script
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Research Contributions

1. **Chain-of-Intent Mechanism**: Novel combination of HMMs and LLMs for contextually aware dialogue generation
2. **MINT-CL Framework**: Multi-task contrastive learning for improved intent classification
3. **MINT-E Corpus**: Multilingual intent-aware e-commerce dialogue dataset
4. **Comprehensive Evaluation**: Extensive experiments across multiple languages and domains

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{liu2025mint,
  title={From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification},
  author={Liu, Junhua and Tan, Yong Keat and Fu, Bin and Lim, Kwan Hui},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Junhua Liu (j@forth.ai)
- Bin Fu (bin.fu@shopee.com)