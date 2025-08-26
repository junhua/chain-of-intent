# Sample Data for Chain-of-Intent Implementation

This directory contains sample data files to enable full reproducibility of the Chain-of-Intent and MINT-CL implementation without requiring access to proprietary datasets.

## Data Structure

```
data/
├── raw/                           # Input data files
│   ├── sample_seed_conversations.json    # Seed conversations for domain knowledge extraction
│   ├── sample_ecommerce_dataset.csv     # Alternative CSV format dataset
│   └── intent_taxonomy.json             # Hierarchical intent taxonomy definition
├── processed/                     # Generated during runtime
│   ├── train_conversations.json         # Training set after splitting
│   ├── val_conversations.json          # Validation set
│   ├── test_conversations.json         # Test set
│   ├── generated_conversations.json    # Chain-of-Intent generated dialogues
│   └── hmm_parameters.json             # Extracted HMM parameters
└── README.md                      # This file
```

## File Descriptions

### 1. `sample_seed_conversations.json`

**Purpose**: Seed conversations used for domain knowledge extraction in Chain-of-Intent pipeline.

**Format**: JSON array of conversation objects
```json
{
  "conversation_id": "seed_001",
  "turns": 2,
  "language": "en", 
  "dialogue": [
    {
      "turn": 1,
      "question": "Where is my order #ORD12345?",
      "intent": "track_order",
      "answer": "Let me check your order status..."
    }
  ]
}
```

**Content**: 
- 10 sample conversations
- 5 languages: English, Indonesian (id), Malaysian (my), Philippine (ph), Singaporean (sg)
- 14 different e-commerce intents
- Realistic customer service interactions

### 2. `sample_ecommerce_dataset.csv`

**Purpose**: Alternative dataset format for users who prefer CSV input.

**Format**: CSV with columns: `conversation_id,turn,question,intent,answer,language,domain`

**Content**:
- 15 conversation turns across 10 conversations
- Multilingual examples (English, Malay, Filipino, Indonesian)
- Common e-commerce customer service scenarios

### 3. `intent_taxonomy.json`

**Purpose**: Defines the hierarchical intent classification structure used in MINT-CL.

**Content**:
- **3-level hierarchy**: Level 1 (broad categories) → Level 2 (sub-categories) → Level 3 (specific intents)
- **Intent definitions**: Clear descriptions of each intent
- **Language examples**: Sample phrases for each intent across multiple languages
- **Mappings**: How intents relate across hierarchy levels

**Hierarchy Structure**:
- **Level 1**: 7 broad categories (order, product, payment, support, account, shipping, general)
- **Level 2**: 21 sub-categories 
- **Level 3**: 60+ specific intents

## Usage

### Quick Start with Sample Data

1. **Use seed conversations for domain knowledge extraction**:
```bash
# The system will automatically detect and use sample_seed_conversations.json
python generate_dialogues.py --config config.yaml --num-conversations 50
```

2. **Train MINT-CL with generated data**:
```bash
# Uses processed conversations automatically
python train.py --config config.yaml
```

3. **Test with CSV format**:
```python
from data_utils import DataProcessor
processor = DataProcessor()
conversations = processor.load_ecommerce_data('data/raw/sample_ecommerce_dataset.csv')
```

### Customizing for Your Domain

1. **Replace seed conversations**: Update `sample_seed_conversations.json` with your domain-specific conversations
2. **Modify intent taxonomy**: Edit `intent_taxonomy.json` to match your classification needs
3. **Add more languages**: Include conversations in additional languages following the same format

## Data Statistics

### Sample Seed Conversations
- **Total Conversations**: 10
- **Total Turns**: 20
- **Languages**: 5 (en, id, my, ph, sg)
- **Unique Intents**: 14
- **Average Turns per Conversation**: 2.0
- **Average Question Length**: 8.5 words

### Intent Distribution
| Intent | Count | Percentage |
|--------|-------|------------|
| track_order | 3 | 15% |
| delivery_time | 3 | 15% |
| return_policy | 2 | 10% |
| product_availability | 2 | 10% |
| Others | 10 | 50% |

### Language Distribution
| Language | Conversations | Percentage |
|----------|---------------|------------|
| English (en) | 5 | 50% |
| Indonesian (id) | 1 | 10% |
| Malaysian (my) | 1 | 10% |
| Philippine (ph) | 1 | 10% |
| Singaporean (sg) | 1 | 10% |
| Multilingual turns | 1 | 10% |

## Data Quality

### Authenticity
- Conversations reflect realistic customer service interactions
- Language usage is natural and colloquial where appropriate
- Intent sequences follow logical conversation flows

### Diversity
- Multiple languages and regional variants
- Various conversation lengths (1-3 turns)
- Different customer service scenarios
- Mix of simple and complex queries

### Completeness
- Every turn has required fields (question, intent, answer)
- Consistent formatting across all files
- Proper language codes and domain labels

## Extending the Dataset

To add your own data while maintaining compatibility:

1. **Follow the JSON structure** for conversation files
2. **Use consistent intent naming** as defined in the taxonomy
3. **Include language codes** using standard ISO codes
4. **Maintain turn ordering** (start from 1, increment sequentially)
5. **Provide meaningful answers** for training quality

## Notes

- This sample data is designed for demonstration and testing purposes
- For production use, replace with your domain-specific data
- The intent taxonomy can be customized for your specific use case
- All sample conversations are synthetic and created for educational purposes