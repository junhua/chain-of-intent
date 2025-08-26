#!/usr/bin/env python3
"""
Dialogue generation script using Chain-of-Intent method.

This script implements the complete pipeline for generating intent-driven
dialogues using the Chain-of-Intent mechanism described in the paper.
"""

import os
import sys
import logging
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import random
import numpy as np

from chain_of_intent import ChainOfIntent
from data_utils import DataProcessor, ConversationData

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
            logging.FileHandler(log_file.replace('.log', '_generation.log')),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_seed_data(config: Dict[str, Any]) -> List[Dict]:
    """
    Load seed conversation data for domain knowledge extraction.
    
    In practice, this would load your existing e-commerce chat logs
    to extract intent transitions and turn distributions.
    """
    logger = logging.getLogger(__name__)
    
    # Check for existing seed data
    raw_data_path = Path(config['data']['raw_data_path'])
    
    # Try sample seed data first, then fallback to generated seed data
    sample_seed_file = raw_data_path / "sample_seed_conversations.json"
    seed_data_file = raw_data_path / "seed_conversations.json"
    
    if sample_seed_file.exists():
        logger.info(f"Loading sample seed data from {sample_seed_file}")
        with open(sample_seed_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif seed_data_file.exists():
        logger.info(f"Loading seed data from {seed_data_file}")
        with open(seed_data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Create sample seed data for demonstration
    logger.info("Creating sample seed data for demonstration...")
    
    sample_seed_data = create_sample_seed_data()
    
    # Save sample data for future use
    raw_data_path.mkdir(parents=True, exist_ok=True)
    with open(seed_data_file, 'w', encoding='utf-8') as f:
        json.dump(sample_seed_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved sample seed data to {seed_data_file}")
    return sample_seed_data

def create_sample_seed_data() -> List[Dict]:
    """Create sample seed conversation data"""
    
    # Define realistic e-commerce intents and transitions
    intents = [
        'track_order', 'delivery_time', 'return_policy', 'product_info',
        'payment_issue', 'cancel_order', 'refund_status', 'size_guide',
        'product_availability', 'discount_inquiry', 'account_issue',
        'shipping_cost', 'order_modification', 'technical_support'
    ]
    
    # Define realistic intent transitions (some intents often follow others)
    intent_transitions = {
        'track_order': ['delivery_time', 'order_modification', 'cancel_order'],
        'delivery_time': ['shipping_cost', 'order_modification'],
        'return_policy': ['refund_status', 'order_modification'],
        'product_info': ['product_availability', 'size_guide', 'discount_inquiry'],
        'payment_issue': ['refund_status', 'account_issue', 'technical_support'],
        'cancel_order': ['refund_status', 'return_policy'],
        'refund_status': ['account_issue', 'payment_issue'],
        'size_guide': ['return_policy', 'product_availability'],
        'product_availability': ['discount_inquiry', 'shipping_cost'],
        'discount_inquiry': ['product_info', 'payment_issue'],
        'account_issue': ['technical_support', 'payment_issue'],
        'shipping_cost': ['delivery_time', 'track_order'],
        'order_modification': ['delivery_time', 'cancel_order'],
        'technical_support': ['account_issue', 'payment_issue']
    }
    
    # Question templates for each intent
    question_templates = {
        'track_order': [
            "Where is my order #{}?",
            "Can you track my package?",
            "What's the status of my order?",
            "I haven't received my order yet"
        ],
        'delivery_time': [
            "When will my order arrive?",
            "How long does shipping take?",
            "What's the expected delivery date?",
            "Can you expedite my delivery?"
        ],
        'return_policy': [
            "What's your return policy?",
            "How can I return this item?",
            "Can I return this without the receipt?",
            "Is there a return deadline?"
        ],
        'product_info': [
            "Tell me more about this product",
            "What are the product specifications?",
            "Is this product genuine?",
            "What materials is this made of?"
        ],
        'payment_issue': [
            "My payment failed",
            "I was charged twice",
            "The payment didn't go through",
            "There's an issue with my billing"
        ],
        'cancel_order': [
            "I want to cancel my order",
            "Can I cancel this purchase?",
            "How do I cancel my order?",
            "I need to cancel my recent order"
        ],
        'refund_status': [
            "Where's my refund?",
            "When will I get my money back?",
            "What's the refund status?",
            "How long for the refund to process?"
        ],
        'size_guide': [
            "What size should I order?",
            "Do you have a size chart?",
            "How does this fit?",
            "Is this true to size?"
        ],
        'product_availability': [
            "Is this item in stock?",
            "When will this be available?",
            "Do you have this in my size?",
            "Can I get notified when it's back?"
        ],
        'discount_inquiry': [
            "Do you have any discounts?",
            "Is there a promo code?",
            "Can I get a better price?",
            "Are there any ongoing sales?"
        ],
        'account_issue': [
            "I can't log into my account",
            "How do I reset my password?",
            "My account is locked",
            "I forgot my login details"
        ],
        'shipping_cost': [
            "How much is shipping?",
            "Is there free delivery?",
            "What are the shipping options?",
            "Can I get free shipping?"
        ],
        'order_modification': [
            "Can I change my order?",
            "I want to modify my purchase",
            "Can I add items to my order?",
            "How do I change the delivery address?"
        ],
        'technical_support': [
            "The website isn't working",
            "I'm having technical issues",
            "The app keeps crashing",
            "I can't complete my purchase"
        ]
    }
    
    # Generate sample conversations
    seed_conversations = []
    
    for conv_id in range(500):  # Generate 500 seed conversations
        # Random conversation length (weighted towards shorter conversations)
        num_turns = int(np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
        
        dialogue = []
        current_intent = random.choice(intents)
        
        for turn in range(num_turns):
            # Generate question
            question_template = random.choice(question_templates[current_intent])
            if '{}' in question_template:
                order_id = f"ORD{random.randint(10000, 99999)}"
                question = question_template.format(order_id)
            else:
                question = question_template
            
            # Generate simple answer
            answer = generate_simple_answer(current_intent)
            
            dialogue.append({
                'turn': turn + 1,
                'question': question,
                'intent': current_intent,
                'answer': answer
            })
            
            # Choose next intent based on transitions (if not last turn)
            if turn < num_turns - 1:
                if current_intent in intent_transitions:
                    # 70% chance to follow transition, 30% random
                    if random.random() < 0.7:
                        current_intent = random.choice(intent_transitions[current_intent])
                    else:
                        current_intent = random.choice(intents)
                else:
                    current_intent = random.choice(intents)
        
        seed_conversations.append({
            'conversation_id': f'seed_{conv_id}',
            'turns': int(num_turns),
            'language': random.choice(['en', 'id', 'my', 'ph', 'sg']),
            'dialogue': dialogue
        })
    
    return seed_conversations

def generate_simple_answer(intent: str) -> str:
    """Generate simple answers for seed data"""
    answer_templates = {
        'track_order': "Let me check your order status for you.",
        'delivery_time': "Your order should arrive within 3-5 business days.",
        'return_policy': "You can return items within 30 days of purchase.",
        'product_info': "This product is made with high-quality materials.",
        'payment_issue': "I'll help you resolve this payment issue.",
        'cancel_order': "I can help you cancel your order.",
        'refund_status': "Your refund is being processed and should appear in 3-5 days.",
        'size_guide': "Please check our size guide for the best fit.",
        'product_availability': "Let me check the availability for you.",
        'discount_inquiry': "We currently have a 10% discount on selected items.",
        'account_issue': "I'll help you resolve your account issue.",
        'shipping_cost': "Shipping costs depend on your location and order value.",
        'order_modification': "I can help you modify your order if it hasn't shipped yet.",
        'technical_support': "I'll connect you with our technical support team."
    }
    
    return answer_templates.get(intent, "I'll help you with that.")

def run_dialogue_generation(config: Dict[str, Any]):
    """Main dialogue generation function"""
    logger = logging.getLogger(__name__)
    
    # Load seed conversation data
    logger.info("Loading seed conversation data...")
    seed_data = load_seed_data(config)
    
    # Convert seed data to format expected by ChainOfIntent
    chat_logs = []
    for conv in seed_data:
        for turn in conv['dialogue']:
            chat_logs.append({
                'conversation_id': conv['conversation_id'],
                'turn': turn['turn'],
                'question': turn['question'],
                'intent': turn['intent'],
                'answer': turn['answer']
            })
    
    logger.info(f"Loaded {len(chat_logs)} turns from {len(seed_data)} seed conversations")
    
    # Initialize Chain-of-Intent
    coi = ChainOfIntent(llm_model=config['chain_of_intent']['primary_llm'])
    
    # Extract domain knowledge
    logger.info("Extracting domain knowledge...")
    intent_transitions = coi.extract_domain_knowledge(chat_logs)
    
    # Save extracted knowledge
    knowledge_path = Path(config['output']['hmm_parameters_path'])
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    
    knowledge_data = {
        'turn_distribution': intent_transitions.turn_distribution,
        'initial_distribution': intent_transitions.initial_distribution,
        'transition_matrix': intent_transitions.transition_matrix,
        'intent_to_questions': dict(coi.intent_to_questions)
    }
    
    with open(knowledge_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved domain knowledge to {knowledge_path}")
    
    # Generate new conversations
    max_conversations = config['chain_of_intent']['max_conversations']
    logger.info(f"Generating {max_conversations} new conversations...")
    
    generated_conversations = []
    batch_size = 50  # Generate in batches to show progress
    
    for batch_start in range(0, max_conversations, batch_size):
        batch_end = min(batch_start + batch_size, max_conversations)
        batch_conversations = []
        
        for i in range(batch_start, batch_end):
            try:
                # Generate conversation
                conversation = coi.generate_conversation()
                conversation['conv_id'] = f"generated_{i}"
                batch_conversations.append(conversation)
                
                # Generate alternative answer for contrastive learning
                if config['chain_of_intent']['enable_answer_ranking']:
                    conversation = coi.generate_alternative_answer(
                        conversation, 
                        config['chain_of_intent']['alternative_llm']
                    )
                
            except Exception as e:
                logger.warning(f"Failed to generate conversation {i}: {e}")
                continue
        
        generated_conversations.extend(batch_conversations)
        logger.info(f"Generated {len(generated_conversations)} conversations so far...")
    
    # Save generated conversations
    output_path = Path(config['output']['generated_conversations_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    coi.save_conversations(generated_conversations, str(output_path))
    logger.info(f"Saved {len(generated_conversations)} generated conversations to {output_path}")
    
    # Convert to ConversationData format for further processing
    processor = DataProcessor()
    processed_conversations = []
    
    for conv in generated_conversations:
        # Convert to ConversationData format
        dialogue_turns = []
        for turn_data in conv['dialogue']:
            dialogue_turn = {
                'turn': turn_data['turn'],
                'question': turn_data['question'],
                'intent': turn_data['intent'],
                'answer': turn_data['answer'],
                'reference_question': turn_data.get('reference_question', ''),
            }
            
            if 'alternative_answer' in turn_data:
                dialogue_turn['alternative_answer'] = turn_data['alternative_answer']
            
            dialogue_turns.append(dialogue_turn)
        
        conv_data = ConversationData(
            conversation_id=conv['conv_id'],
            turns=conv['turns'],
            language='en',  # Default language for generated conversations
            domain='ecommerce',
            dialogue=dialogue_turns,
            metadata={'generated': True, 'intents': conv['intents']}
        )
        
        processed_conversations.append(conv_data)
    
    # Save processed conversations for training
    processed_path = Path(config['data']['processed_data_path']) / "generated_conversations.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_processed_data(processed_conversations, str(processed_path))
    
    # Generate statistics
    stats = processor.extract_intent_statistics(processed_conversations)
    
    logger.info("Generation Statistics:")
    logger.info(f"Total Conversations: {stats['total_conversations']}")
    logger.info(f"Total Turns: {stats['total_turns']}")
    logger.info(f"Average Turns per Conversation: {stats['avg_turns_per_conversation']:.2f}")
    logger.info(f"Average Question Length: {stats['avg_question_length']:.2f} words")
    logger.info(f"Unique Intents: {len(stats['intents'])}")
    
    # Save statistics
    stats_path = Path(config['output']['results_dir']) / "generation_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Saved generation statistics to {stats_path}")
    logger.info("Dialogue generation completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate dialogues using Chain-of-Intent')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-conversations', type=int, default=None,
                        help='Number of conversations to generate (overrides config)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output path for generated conversations (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_conversations:
        config['chain_of_intent']['max_conversations'] = args.num_conversations
    
    if args.output_path:
        config['output']['generated_conversations_path'] = args.output_path
    
    if args.debug:
        config['development']['debug'] = True
        config['logging']['level'] = 'DEBUG'
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting dialogue generation with config: {args.config}")
    logger.info(f"Will generate {config['chain_of_intent']['max_conversations']} conversations")
    
    # Set random seed for reproducibility
    if config['experiment']['set_seed']:
        set_seed(config['data']['random_seed'])
        logger.info(f"Random seed set to: {config['data']['random_seed']}")
    
    try:
        # Run dialogue generation
        run_dialogue_generation(config)
        
    except Exception as e:
        logger.error(f"Dialogue generation failed with error: {e}")
        if config['development']['debug']:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()