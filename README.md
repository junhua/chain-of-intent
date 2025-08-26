# Chain-of-Intent: Intent-Driven Dialogue Generation with Multi-Task Contrastive Learning

This repository contains a source code implementation for the paper "From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification" accepted at CIKM 2025.

## Important Note

The original implementation code cannot be shared due to copyright and proprietary restrictions. This repository provides a clean implementation generated based on the methodologies and algorithms described in the published paper.

## Dataset Availability

The MINT-E dataset will be made available at: https://huggingface.co/datasets/fubincom/MINT_E/tree/main (currently under construction)

## Pipeline Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Domain Knowledge│    │ Chain-of-Intent  │    │    MINT-CL      │
│   Extraction     │───▶│   Generation     │───▶│ Classification  │
└──────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
    ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
    │Chat Logs│              │HMM+LLMs │              │XLM-R +  │
    │P(T),P(I)│              │Intent   │              │Multi-   │
    │Transi-  │              │Sampling │              │Task     │
    │tions    │              │Dialogue │              │Learning │
    └─────────┘              │Generate │              └─────────┘
                             └─────────┘
```

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

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- CUDA-capable GPU (recommended for training)
- OpenAI API key (for dialogue generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/junhua/chain-of-intent.git
cd chain-of-intent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key (required for Chain-of-Intent dialogue generation):
```bash
export OPENAI_API_KEY="your-openai-api-key"
# Or add to your .bashrc/.zshrc for persistence
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.bashrc
```

## Quick Start

### Step 1: Sample Data (Ready to Use)

The repository includes sample data for immediate testing and reproduction:

```bash
# Sample data is already included in the repository
ls data/raw/
# sample_seed_conversations.json    - 10 multilingual e-commerce conversations  
# sample_ecommerce_dataset.csv      - Alternative CSV format
# intent_taxonomy.json              - Hierarchical intent definitions
```

**No additional data preparation needed!** The system will automatically use the sample data.

### Step 2: Generate Dialogues

Generate intent-driven conversations using Chain-of-Intent:

```bash
# Generate 1000 conversations (adjust number as needed)
python generate_dialogues.py --config config.yaml --num-conversations 1000

# Optional: specify custom output path
python generate_dialogues.py --config config.yaml --output-path data/my_conversations.json
```

**Output**: Generated conversations will be saved to `data/generated/conversations.json` and processed data to `data/processed/`.

### Step 3: Train MINT-CL Model

Train the multi-task contrastive learning model:

```bash
# Train with default configuration
python train.py --config config.yaml

# Optional: enable debug mode for development
python train.py --config config.yaml --debug
```

**Output**: 
- Model checkpoints saved to `models/`
- Training metrics saved to `results/training_history.json`
- Final evaluation results in `results/final_metrics.json`

### Step 4: Monitor Training

Check training progress:
```bash
# View training logs
tail -f logs/training.log

# If using TensorBoard (enabled in config.yaml)
tensorboard --logdir runs/
```

### Step 5: Verify Installation

Run the basic functionality test to ensure everything works:

```bash
python test_basic_functionality.py
```

**Expected Output**: All tests should pass with "🎉 All tests passed!"

### Step 6: Customize Configuration

Edit `config.yaml` to customize:

```yaml
# Key parameters to modify
chain_of_intent:
  max_conversations: 5000        # Number of dialogues to generate
  primary_llm: "gpt-3.5-turbo"  # LLM for generation

mint_cl:
  batch_size: 16                 # Adjust based on GPU memory
  learning_rate: 2e-5            # Learning rate
  num_epochs: 10                 # Training epochs

data:
  languages: ["en", "id", "my"]  # Languages to support
  train_ratio: 0.7               # Train/val/test split ratios
```

## Data Format

### Sample Data Included

The repository comes with ready-to-use sample data:

- **`data/raw/sample_seed_conversations.json`** - 10 multilingual conversations for domain knowledge extraction
- **`data/raw/sample_ecommerce_dataset.csv`** - Alternative CSV format with 15 conversation turns  
- **`data/raw/intent_taxonomy.json`** - Complete hierarchical intent taxonomy
- **`data/README.md`** - Detailed data documentation

### Custom Data Format

**For your own seed conversations**, use this JSON format:

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

### Supported File Formats

- **JSON**: `data.json` (recommended)
- **CSV**: `data.csv` with columns: `conversation_id`, `turn`, `question`, `intent`, `answer`, `language`
- **Pickle**: `data.pkl` (for pre-processed ConversationData objects)

### Intent Hierarchy

The system automatically builds a 3-level hierarchical intent classification:
- **Level 1**: Broad categories (e.g., "order", "product", "payment")
- **Level 2**: Sub-categories (e.g., "order_tracking", "product_info")  
- **Level 3**: Specific intents (e.g., "track_order", "delivery_time")

**Note**: Intent hierarchy is auto-generated from your data. For custom hierarchies, modify the `build_intent_hierarchy()` function in `mint_cl.py`.

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
├── chain_of_intent.py           # Chain-of-Intent implementation
├── mint_cl.py                  # MINT-CL model and training
├── data_utils.py               # Data processing utilities
├── train.py                    # Training script
├── generate_dialogues.py       # Dialogue generation script
├── test_basic_functionality.py # Basic functionality tests
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                  # This file
└── data/                      # Sample data for reproducibility
    ├── raw/                   # Input sample data
    │   ├── sample_seed_conversations.json
    │   ├── sample_ecommerce_dataset.csv
    │   └── intent_taxonomy.json
    ├── processed/            # Generated during runtime
    └── README.md            # Data documentation
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

For questions or issues, please create an issue in the repo.
