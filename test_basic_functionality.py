#!/usr/bin/env python3
"""
Basic functionality test for the Chain-of-Intent implementation.

This script tests core functionality without requiring extensive training
or API keys, providing a quick verification that the implementation works.
"""

import sys
import logging
from pathlib import Path
import json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        from chain_of_intent import ChainOfIntent
        from mint_cl import MINTContrastiveClassifier, build_intent_hierarchy
        from data_utils import DataProcessor, ConversationData
        logger.info("✓ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    try:
        from data_utils import DataProcessor, ConversationData
        
        # Create sample conversation data (multiple conversations for splitting test)
        sample_conversations = []
        for i in range(10):  # Create 10 conversations for proper splitting
            sample_conversations.append(ConversationData(
                conversation_id=f'test_{i}',
                turns=2,
                language='en',
                domain='ecommerce',
                dialogue=[
                    {
                        'turn': 1,
                        'question': f'Where is my order {i}?',
                        'intent': 'track_order',
                        'answer': 'Let me check that for you.'
                    },
                    {
                        'turn': 2,
                        'question': 'When will it arrive?',
                        'intent': 'delivery_time',
                        'answer': 'It should arrive tomorrow.'
                    }
                ]
            ))
        
        processor = DataProcessor()
        
        # Test validation
        validation_report = processor.validate_conversation_data(sample_conversations)
        assert validation_report['validation_success'], "Validation should pass"
        
        # Test statistics extraction
        stats = processor.extract_intent_statistics(sample_conversations)
        assert stats['total_conversations'] == 10, "Should have 10 conversations"
        assert stats['total_turns'] == 20, "Should have 20 turns"
        
        # Test train/val/test split
        dataset_split = processor.create_train_val_test_split(
            sample_conversations, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert len(dataset_split.train) + len(dataset_split.validation) + len(dataset_split.test) == 10
        
        logger.info("✓ Data processing tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data processing test failed: {e}")
        return False

def test_intent_hierarchy():
    """Test intent hierarchy building"""
    try:
        from mint_cl import build_intent_hierarchy
        
        # Sample conversation data
        conversations = [
            {
                'dialogue': [
                    {'intent': 'track_order'},
                    {'intent': 'delivery_time'},
                    {'intent': 'return_policy'}
                ]
            }
        ]
        
        hierarchy = build_intent_hierarchy(conversations)
        
        assert len(hierarchy.level3_to_idx) == 3, "Should have 3 level-3 intents"
        assert 'track_order' in hierarchy.level3_to_idx, "Should contain track_order intent"
        
        logger.info("✓ Intent hierarchy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Intent hierarchy test failed: {e}")
        return False

def test_chain_of_intent_basic():
    """Test Chain-of-Intent basic functionality without API calls"""
    try:
        from chain_of_intent import ChainOfIntent
        
        # Sample chat logs for domain knowledge extraction
        sample_logs = [
            {
                'conversation_id': 'test_1',
                'turn': 1,
                'question': 'Where is my order?',
                'intent': 'track_order',
                'answer': 'Let me check that.'
            },
            {
                'conversation_id': 'test_1', 
                'turn': 2,
                'question': 'When will it arrive?',
                'intent': 'delivery_time',
                'answer': 'It should arrive soon.'
            }
        ]
        
        coi = ChainOfIntent()
        
        # Test domain knowledge extraction
        intent_transitions = coi.extract_domain_knowledge(sample_logs)
        
        assert len(intent_transitions.initial_distribution) > 0, "Should have initial distribution"
        assert len(intent_transitions.turn_distribution) > 0, "Should have turn distribution"
        
        # Test intent sequence sampling
        turns, intents = coi.sample_intent_sequence()
        assert turns > 0, "Should sample positive number of turns"
        assert len(intents) == turns, "Intent sequence length should match turns"
        
        logger.info("✓ Chain-of-Intent basic tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Chain-of-Intent test failed: {e}")
        return False

def test_model_initialization():
    """Test MINT-CL model initialization"""
    try:
        from mint_cl import MINTContrastiveClassifier, build_intent_hierarchy
        
        # Create simple hierarchy
        conversations = [
            {
                'dialogue': [
                    {'intent': 'track_order'},
                    {'intent': 'delivery_time'}
                ]
            }
        ]
        
        hierarchy = build_intent_hierarchy(conversations)
        
        # Initialize model
        model = MINTContrastiveClassifier(hierarchy, hidden_dim=768)
        
        # Check model components
        assert hasattr(model, 'intent_classifier'), "Should have intent classifier"
        assert hasattr(model, 'response_ranker'), "Should have response ranker"
        
        logger.info("✓ Model initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model initialization test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        import yaml
        
        # Check if config file exists and is valid
        config_path = Path('config.yaml')
        if not config_path.exists():
            logger.error("✗ config.yaml not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['chain_of_intent', 'mint_cl', 'data', 'output']
        for section in required_sections:
            assert section in config, f"Config should have {section} section"
        
        logger.info("✓ Configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all basic functionality tests"""
    logger.info("Running Chain-of-Intent basic functionality tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Processing Test", test_data_processing),
        ("Intent Hierarchy Test", test_intent_hierarchy),
        ("Chain-of-Intent Basic Test", test_chain_of_intent_basic),
        ("Model Initialization Test", test_model_initialization),
        ("Configuration Test", test_configuration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        logger.info("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)