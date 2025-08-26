#!/usr/bin/env python3
"""
Training script for MINT-CL multi-turn intent classification model.

This script implements the complete training pipeline for the MINT-CL framework
described in the paper, including data loading, model initialization, training,
and evaluation.
"""

import os
import sys
import logging
import argparse
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json

from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Import our modules
from data_utils import DataProcessor, ConversationData
from mint_cl import (
    MINTContrastiveClassifier, 
    MINTTrainer, 
    MultiTurnDataset,
    build_intent_hierarchy
)
from chain_of_intent import ChainOfIntent

def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    
    # Create logs directory
    log_file = config['logging']['log_file']
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config: Dict[str, Any]) -> tuple:
    """
    Prepare data for training.
    
    Returns:
        Tuple of (train_conversations, val_conversations, test_conversations, processor)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Check if processed data exists
    processed_path = Path(config['data']['processed_data_path'])
    train_path = processed_path / "train_conversations.json"
    val_path = processed_path / "val_conversations.json"
    test_path = processed_path / "test_conversations.json"
    
    if all(path.exists() for path in [train_path, val_path, test_path]):
        logger.info("Loading existing processed data...")
        
        train_conversations = processor.load_ecommerce_data(str(train_path))
        val_conversations = processor.load_ecommerce_data(str(val_path))
        test_conversations = processor.load_ecommerce_data(str(test_path))
        
    else:
        logger.info("Processing raw data...")
        
        # Load raw data (this would be your actual data loading)
        raw_data_path = config['data']['raw_data_path']
        
        # For demonstration, create sample data
        # In practice, you would load from your actual dataset
        sample_conversations = create_sample_data()
        
        # Create train/val/test split
        dataset_split = processor.create_train_val_test_split(
            sample_conversations,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            random_state=config['data']['random_seed']
        )
        
        train_conversations = dataset_split.train
        val_conversations = dataset_split.validation
        test_conversations = dataset_split.test
        
        # Save processed data
        processed_path.mkdir(parents=True, exist_ok=True)
        processor.save_processed_data(train_conversations, str(train_path))
        processor.save_processed_data(val_conversations, str(val_path))
        processor.save_processed_data(test_conversations, str(test_path))
        
        logger.info(f"Saved processed data: Train={len(train_conversations)}, "
                   f"Val={len(val_conversations)}, Test={len(test_conversations)}")
    
    # Augment with alternative responses for contrastive learning
    if config['mint_cl']['augment_with_alternatives']:
        logger.info("Augmenting data with alternative responses...")
        train_conversations = processor.augment_conversations_with_alternatives(train_conversations)
        val_conversations = processor.augment_conversations_with_alternatives(val_conversations)
    
    return train_conversations, val_conversations, test_conversations, processor

def create_sample_data() -> list:
    """Create sample conversation data for demonstration"""
    sample_conversations = []
    
    intents = ['track_order', 'delivery_time', 'return_policy', 'product_info', 
               'payment_issue', 'cancel_order', 'refund_status']
    
    for i in range(100):  # Create 100 sample conversations
        num_turns = random.randint(1, 5)
        dialogue = []
        
        for turn in range(num_turns):
            intent = random.choice(intents)
            
            # Sample questions based on intent
            question_templates = {
                'track_order': ["Where is my order?", "Can you track my package?", "Order status please"],
                'delivery_time': ["When will it arrive?", "Delivery time?", "How long for shipping?"],
                'return_policy': ["How to return?", "Return policy?", "Can I return this?"],
                'product_info': ["Tell me about this product", "Product details?", "Is this available?"],
                'payment_issue': ["Payment failed", "Billing problem", "Card was charged twice"],
                'cancel_order': ["Cancel my order", "I want to cancel", "How to cancel?"],
                'refund_status': ["Refund status?", "Where is my refund?", "When will I get refund?"]
            }
            
            question = random.choice(question_templates[intent])
            answer = f"I'll help you with {intent.replace('_', ' ')}."
            
            dialogue.append({
                'turn': turn + 1,
                'question': question,
                'intent': intent,
                'answer': answer
            })
        
        conv = ConversationData(
            conversation_id=f"conv_{i}",
            turns=num_turns,
            language=random.choice(['en', 'id', 'my']),
            domain='ecommerce',
            dialogue=dialogue
        )
        
        sample_conversations.append(conv)
    
    return sample_conversations

def initialize_model(config: Dict[str, Any], hierarchy) -> MINTContrastiveClassifier:
    """Initialize the MINT-CL model"""
    model = MINTContrastiveClassifier(
        hierarchy=hierarchy,
        hidden_dim=config['mint_cl']['hidden_dim']
    )
    
    # Set loss weights
    model.intent_weight = config['mint_cl']['intent_loss_weight']
    model.contrastive_weight = config['mint_cl']['contrastive_loss_weight']
    
    return model

def create_data_loaders(conversations_data: tuple, hierarchy, config: Dict[str, Any]) -> tuple:
    """Create data loaders for training, validation, and testing"""
    train_conversations, val_conversations, test_conversations, _ = conversations_data
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['mint_cl']['base_model'])
    
    # Create datasets
    train_dataset = MultiTurnDataset(
        conversations=[conv.__dict__ for conv in train_conversations],
        tokenizer=tokenizer,
        hierarchy=hierarchy,
        max_length=config['mint_cl']['max_sequence_length'],
        include_responses=config['mint_cl']['include_response_ranking']
    )
    
    val_dataset = MultiTurnDataset(
        conversations=[conv.__dict__ for conv in val_conversations],
        tokenizer=tokenizer,
        hierarchy=hierarchy,
        max_length=config['mint_cl']['max_sequence_length'],
        include_responses=config['mint_cl']['include_response_ranking']
    )
    
    test_dataset = MultiTurnDataset(
        conversations=[conv.__dict__ for conv in test_conversations],
        tokenizer=tokenizer,
        hierarchy=hierarchy,
        max_length=config['mint_cl']['max_sequence_length'],
        include_responses=False  # No contrastive learning for test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mint_cl']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['mint_cl']['eval_batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['mint_cl']['eval_batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader, tokenizer

def run_training(config: Dict[str, Any]):
    """Main training function"""
    logger = logging.getLogger(__name__)
    
    # Set device
    if config['hardware']['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['hardware']['device']
    
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    conversations_data = prepare_data(config)
    train_conversations, val_conversations, test_conversations, processor = conversations_data
    
    # Build intent hierarchy
    logger.info("Building intent hierarchy...")
    all_conversations = train_conversations + val_conversations + test_conversations
    hierarchy = build_intent_hierarchy([conv.__dict__ for conv in all_conversations])
    
    logger.info(f"Intent hierarchy: Level1={len(hierarchy.level1_to_idx)}, "
               f"Level2={len(hierarchy.level2_to_idx)}, Level3={len(hierarchy.level3_to_idx)}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        conversations_data, hierarchy, config
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = initialize_model(config, hierarchy)
    
    # Initialize trainer
    trainer = MINTTrainer(model, tokenizer, device=device)
    
    # Training
    logger.info("Starting training...")
    training_history = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=config['mint_cl']['num_epochs'],
        learning_rate=config['mint_cl']['learning_rate']
    )
    
    # Save training history
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'training_history.json', 'w') as f:
        # Convert any tensor values to floats for JSON serialization
        serializable_history = []
        for epoch_data in training_history:
            epoch_dict = {
                'epoch': epoch_data['epoch'],
                'train_metrics': {k: float(v) if torch.is_tensor(v) else v 
                                for k, v in epoch_data['train_metrics'].items()},
                'val_metrics': {k: float(v) if torch.is_tensor(v) else v 
                              for k, v in epoch_data['val_metrics'].items()}
            }
            serializable_history.append(epoch_dict)
        
        json.dump(serializable_history, f, indent=2)
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    logger.info("Test Results:")
    logger.info(f"Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")
    logger.info(f"Level 1 Accuracy: {test_metrics['level1_accuracy']:.4f}")
    logger.info(f"Level 2 Accuracy: {test_metrics['level2_accuracy']:.4f}")
    logger.info(f"Level 3 Accuracy: {test_metrics['level3_accuracy']:.4f}")
    
    # Save final metrics
    final_metrics = {k: float(v) if torch.is_tensor(v) else v 
                    for k, v in test_metrics.items()}
    
    with open(results_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save model
    model_dir = Path(config['output']['model_save_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir / 'final_model.pt'))
    
    logger.info("Training completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MINT-CL model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.debug:
        config['development']['debug'] = True
        config['logging']['level'] = 'DEBUG'
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with config: {args.config}")
    
    # Set random seed for reproducibility
    if config['experiment']['set_seed']:
        set_seed(config['data']['random_seed'])
        logger.info(f"Random seed set to: {config['data']['random_seed']}")
    
    try:
        # Run training
        run_training(config)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if config['development']['debug']:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()