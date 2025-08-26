import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@dataclass
class ConversationData:
    """Data structure for conversation information"""
    conversation_id: str
    turns: int
    language: str
    domain: str
    dialogue: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

@dataclass
class DatasetSplit:
    """Data structure for dataset splits"""
    train: List[ConversationData]
    validation: List[ConversationData] 
    test: List[ConversationData]

class DataProcessor:
    """
    Utility class for processing and preparing data for Chain-of-Intent
    and MINT-CL training.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.intent_encoder = LabelEncoder()
        self.language_encoder = LabelEncoder()
        
    def load_ecommerce_data(self, filepath: str) -> List[ConversationData]:
        """
        Load e-commerce conversation data from various formats.
        
        Args:
            filepath: Path to the data file (JSON, CSV, or pickle)
            
        Returns:
            List of ConversationData objects
        """
        path = Path(filepath)
        
        if path.suffix.lower() == '.json':
            return self._load_json_data(filepath)
        elif path.suffix.lower() == '.csv':
            return self._load_csv_data(filepath)
        elif path.suffix.lower() == '.pkl':
            return self._load_pickle_data(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _load_json_data(self, filepath: str) -> List[ConversationData]:
        """Load data from JSON format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of conversations
            for item in data:
                conv = self._parse_conversation_dict(item)
                if conv:
                    conversations.append(conv)
        elif isinstance(data, dict):
            # Single conversation or nested structure
            if 'conversations' in data:
                for item in data['conversations']:
                    conv = self._parse_conversation_dict(item)
                    if conv:
                        conversations.append(conv)
            else:
                conv = self._parse_conversation_dict(data)
                if conv:
                    conversations.append(conv)
        
        self.logger.info(f"Loaded {len(conversations)} conversations from JSON")
        return conversations
    
    def _load_csv_data(self, filepath: str) -> List[ConversationData]:
        """Load data from CSV format"""
        df = pd.read_csv(filepath)
        
        # Expected columns: conversation_id, turn, question, intent, answer, language, etc.
        required_cols = ['conversation_id', 'question', 'intent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        conversations = {}
        
        for _, row in df.iterrows():
            conv_id = str(row['conversation_id'])
            
            if conv_id not in conversations:
                conversations[conv_id] = {
                    'conversation_id': conv_id,
                    'language': row.get('language', 'unknown'),
                    'domain': row.get('domain', 'ecommerce'),
                    'dialogue': []
                }
            
            turn_data = {
                'turn': row.get('turn', len(conversations[conv_id]['dialogue']) + 1),
                'question': row['question'],
                'intent': row['intent'],
                'answer': row.get('answer', ''),
            }
            
            # Add optional fields
            for col in ['alternative_answer', 'quality_score', 'reference_question']:
                if col in row and pd.notna(row[col]):
                    turn_data[col] = row[col]
            
            conversations[conv_id]['dialogue'].append(turn_data)
        
        # Convert to ConversationData objects
        conv_list = []
        for conv_data in conversations.values():
            # Sort dialogue by turn number
            conv_data['dialogue'].sort(key=lambda x: x.get('turn', 0))
            conv_data['turns'] = len(conv_data['dialogue'])
            
            conv_obj = ConversationData(**conv_data)
            conv_list.append(conv_obj)
        
        self.logger.info(f"Loaded {len(conv_list)} conversations from CSV")
        return conv_list
    
    def _load_pickle_data(self, filepath: str) -> List[ConversationData]:
        """Load data from pickle format"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and all(isinstance(item, ConversationData) for item in data):
            return data
        
        # Convert if needed
        conversations = []
        for item in data:
            if isinstance(item, dict):
                conv = self._parse_conversation_dict(item)
                if conv:
                    conversations.append(conv)
        
        return conversations
    
    def _parse_conversation_dict(self, conv_dict: Dict) -> Optional[ConversationData]:
        """Parse a conversation dictionary into ConversationData"""
        try:
            # Handle different conversation formats
            if 'dialogue' in conv_dict:
                dialogue = conv_dict['dialogue']
            elif 'turns' in conv_dict:
                dialogue = conv_dict['turns']
            elif 'messages' in conv_dict:
                dialogue = conv_dict['messages']
            else:
                # Single turn format
                dialogue = [conv_dict]
            
            # Ensure each turn has required fields
            processed_dialogue = []
            for i, turn in enumerate(dialogue):
                if isinstance(turn, dict):
                    processed_turn = {
                        'turn': turn.get('turn', i + 1),
                        'question': turn.get('question', turn.get('user_message', turn.get('query', ''))),
                        'intent': turn.get('intent', 'unknown'),
                        'answer': turn.get('answer', turn.get('agent_response', turn.get('response', ''))),
                    }
                    
                    # Add optional fields
                    for field in ['alternative_answer', 'quality_score', 'reference_question']:
                        if field in turn:
                            processed_turn[field] = turn[field]
                    
                    processed_dialogue.append(processed_turn)
            
            if not processed_dialogue:
                return None
            
            return ConversationData(
                conversation_id=conv_dict.get('conversation_id', conv_dict.get('conv_id', str(len(processed_dialogue)))),
                turns=len(processed_dialogue),
                language=conv_dict.get('language', conv_dict.get('lang', 'unknown')),
                domain=conv_dict.get('domain', 'ecommerce'),
                dialogue=processed_dialogue,
                metadata=conv_dict.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse conversation: {e}")
            return None
    
    def create_train_val_test_split(self, conversations: List[ConversationData],
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_state: int = 42) -> DatasetSplit:
        """
        Split conversations into train/validation/test sets.
        
        Args:
            conversations: List of conversation data
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed
            
        Returns:
            DatasetSplit object
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Stratify by language if available
        languages = [conv.language for conv in conversations]
        
        if len(set(languages)) > 1:
            # Stratified split
            train_val, test = train_test_split(
                conversations, test_size=test_ratio, 
                stratify=languages, random_state=random_state
            )
            
            train_val_languages = [conv.language for conv in train_val]
            train, val = train_test_split(
                train_val, test_size=val_ratio/(train_ratio + val_ratio),
                stratify=train_val_languages, random_state=random_state
            )
        else:
            # Simple random split
            train_val, test = train_test_split(
                conversations, test_size=test_ratio, random_state=random_state
            )
            
            train, val = train_test_split(
                train_val, test_size=val_ratio/(train_ratio + val_ratio),
                random_state=random_state
            )
        
        self.logger.info(f"Dataset split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return DatasetSplit(train=train, validation=val, test=test)
    
    def extract_intent_statistics(self, conversations: List[ConversationData]) -> Dict[str, Any]:
        """
        Extract comprehensive statistics from conversation data.
        
        Args:
            conversations: List of conversation data
            
        Returns:
            Dictionary with various statistics
        """
        stats = {
            'total_conversations': len(conversations),
            'total_turns': 0,
            'languages': Counter(),
            'domains': Counter(),
            'intents': Counter(),
            'turn_distribution': Counter(),
            'questions_per_intent': defaultdict(list),
            'avg_question_length': 0,
            'avg_turns_per_conversation': 0,
        }
        
        all_questions = []
        
        for conv in conversations:
            stats['total_turns'] += conv.turns
            stats['languages'][conv.language] += 1
            stats['domains'][conv.domain] += 1
            stats['turn_distribution'][conv.turns] += 1
            
            for turn in conv.dialogue:
                intent = turn['intent']
                question = turn['question']
                
                stats['intents'][intent] += 1
                stats['questions_per_intent'][intent].append(question)
                all_questions.append(question)
        
        # Calculate averages
        if all_questions:
            avg_len = np.mean([len(q.split()) for q in all_questions])
            stats['avg_question_length'] = avg_len
        
        if conversations:
            stats['avg_turns_per_conversation'] = stats['total_turns'] / len(conversations)
        
        # Convert Counters to regular dicts for JSON serialization
        stats['languages'] = dict(stats['languages'])
        stats['domains'] = dict(stats['domains'])
        stats['intents'] = dict(stats['intents'])
        stats['turn_distribution'] = dict(stats['turn_distribution'])
        
        return stats
    
    def prepare_hmm_training_data(self, conversations: List[ConversationData]) -> Dict[str, Any]:
        """
        Prepare data for HMM training in Chain-of-Intent.
        
        Args:
            conversations: List of conversation data
            
        Returns:
            Dictionary with HMM training parameters
        """
        # Turn distribution P(T)
        turn_counts = Counter([conv.turns for conv in conversations])
        total_conversations = len(conversations)
        turn_distribution = {turns: count/total_conversations 
                           for turns, count in turn_counts.items()}
        
        # Initial intent distribution P(I_1)
        initial_intents = []
        for conv in conversations:
            if conv.dialogue:
                initial_intents.append(conv.dialogue[0]['intent'])
        
        initial_intent_counts = Counter(initial_intents)
        initial_distribution = {intent: count/len(initial_intents) 
                              for intent, count in initial_intent_counts.items()}
        
        # Intent transition matrix P(I_t|I_{t-1})
        transition_counts = defaultdict(Counter)
        
        for conv in conversations:
            for i in range(1, len(conv.dialogue)):
                prev_intent = conv.dialogue[i-1]['intent']
                curr_intent = conv.dialogue[i]['intent']
                transition_counts[prev_intent][curr_intent] += 1
        
        # Normalize transition probabilities
        transition_matrix = {}
        for prev_intent, next_intents in transition_counts.items():
            total = sum(next_intents.values())
            transition_matrix[prev_intent] = {
                intent: count/total for intent, count in next_intents.items()
            }
        
        # Intent to questions mapping
        intent_to_questions = defaultdict(list)
        for conv in conversations:
            for turn in conv.dialogue:
                intent_to_questions[turn['intent']].append(turn['question'])
        
        return {
            'turn_distribution': turn_distribution,
            'initial_distribution': initial_distribution,
            'transition_matrix': transition_matrix,
            'intent_to_questions': dict(intent_to_questions)
        }
    
    def create_multilingual_splits(self, conversations: List[ConversationData]) -> Dict[str, DatasetSplit]:
        """
        Create language-specific dataset splits for multilingual evaluation.
        
        Args:
            conversations: List of conversation data
            
        Returns:
            Dictionary mapping language codes to DatasetSplit objects
        """
        # Group conversations by language
        lang_conversations = defaultdict(list)
        for conv in conversations:
            lang_conversations[conv.language].append(conv)
        
        multilingual_splits = {}
        
        for language, lang_convs in lang_conversations.items():
            if len(lang_convs) >= 10:  # Minimum conversations for meaningful split
                split = self.create_train_val_test_split(lang_convs)
                multilingual_splits[language] = split
                self.logger.info(f"Created split for {language}: "
                               f"Train={len(split.train)}, Val={len(split.validation)}, Test={len(split.test)}")
            else:
                self.logger.warning(f"Skipping {language} - insufficient data ({len(lang_convs)} conversations)")
        
        return multilingual_splits
    
    def augment_conversations_with_alternatives(self, conversations: List[ConversationData],
                                              alternative_model_responses: Dict[str, str] = None) -> List[ConversationData]:
        """
        Augment conversations with alternative responses for contrastive learning.
        
        Args:
            conversations: Original conversations
            alternative_model_responses: Optional pre-generated alternatives
            
        Returns:
            Augmented conversations with alternative responses
        """
        augmented = []
        
        for conv in conversations:
            new_dialogue = []
            
            for turn in conv.dialogue:
                new_turn = turn.copy()
                
                # Add alternative response if not present
                if 'alternative_answer' not in turn:
                    conv_turn_key = f"{conv.conversation_id}_{turn['turn']}"
                    
                    if alternative_model_responses and conv_turn_key in alternative_model_responses:
                        new_turn['alternative_answer'] = alternative_model_responses[conv_turn_key]
                    else:
                        # Generate simple alternative (in practice, use different model)
                        original = turn.get('answer', '')
                        alternative = self._generate_simple_alternative(original)
                        new_turn['alternative_answer'] = alternative
                
                new_dialogue.append(new_turn)
            
            augmented_conv = ConversationData(
                conversation_id=conv.conversation_id,
                turns=conv.turns,
                language=conv.language,
                domain=conv.domain,
                dialogue=new_dialogue,
                metadata=conv.metadata
            )
            
            augmented.append(augmented_conv)
        
        return augmented
    
    def _generate_simple_alternative(self, original_response: str) -> str:
        """Generate a simple alternative response (placeholder implementation)"""
        if not original_response:
            return "I can help you with that."
        
        # Simple transformations
        alternatives = [
            original_response.replace("I'll", "I will"),
            original_response.replace("can't", "cannot"),
            f"Let me assist you. {original_response}",
            f"Sure, {original_response.lower()}",
            original_response.replace(".", "!"),
        ]
        
        # Return different alternative
        return random.choice([alt for alt in alternatives if alt != original_response])
    
    def save_processed_data(self, data: Any, filepath: str, format: str = 'json'):
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            filepath: Output file path
            format: Output format ('json', 'pickle', 'csv')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, list) and all(isinstance(item, ConversationData) for item in data):
                    # Convert ConversationData objects to dictionaries
                    json_data = [asdict(conv) for conv in data]
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        elif format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        elif format.lower() == 'csv':
            if isinstance(data, list) and all(isinstance(item, ConversationData) for item in data):
                # Convert to flat CSV format
                rows = []
                for conv in data:
                    for turn in conv.dialogue:
                        row = {
                            'conversation_id': conv.conversation_id,
                            'language': conv.language,
                            'domain': conv.domain,
                            'turn': turn['turn'],
                            'question': turn['question'],
                            'intent': turn['intent'],
                            'answer': turn.get('answer', ''),
                        }
                        
                        # Add optional fields
                        for field in ['alternative_answer', 'quality_score', 'reference_question']:
                            row[field] = turn.get(field, '')
                        
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False, encoding='utf-8')
            else:
                raise ValueError("CSV format only supported for ConversationData lists")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved data to {filepath} in {format} format")
    
    def validate_conversation_data(self, conversations: List[ConversationData]) -> Dict[str, Any]:
        """
        Validate conversation data and return validation report.
        
        Args:
            conversations: List of conversations to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_conversations': len(conversations),
            'valid_conversations': 0,
            'issues': [],
            'warnings': []
        }
        
        for i, conv in enumerate(conversations):
            conv_issues = []
            
            # Check required fields
            if not conv.conversation_id:
                conv_issues.append(f"Conversation {i}: Missing conversation_id")
            
            if not conv.dialogue:
                conv_issues.append(f"Conversation {i}: Empty dialogue")
                continue
            
            # Check dialogue structure
            for j, turn in enumerate(conv.dialogue):
                if not turn.get('question'):
                    conv_issues.append(f"Conversation {i}, Turn {j}: Missing question")
                
                if not turn.get('intent'):
                    conv_issues.append(f"Conversation {i}, Turn {j}: Missing intent")
                
                # Check turn numbering
                expected_turn = j + 1
                if turn.get('turn', expected_turn) != expected_turn:
                    report['warnings'].append(f"Conversation {i}, Turn {j}: Turn numbering mismatch")
            
            if not conv_issues:
                report['valid_conversations'] += 1
            else:
                report['issues'].extend(conv_issues)
        
        report['validation_success'] = len(report['issues']) == 0
        report['valid_ratio'] = report['valid_conversations'] / len(conversations) if conversations else 0
        
        return report

def main():
    """Example usage of DataProcessor"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Example: Load and process data
    print("Data processing utilities ready!")
    
    # Create sample data for demonstration
    sample_conversations = [
        ConversationData(
            conversation_id='conv_1',
            turns=2,
            language='en',
            domain='ecommerce',
            dialogue=[
                {
                    'turn': 1,
                    'question': 'Where is my order?',
                    'intent': 'track_order',
                    'answer': 'Let me check your order status.'
                },
                {
                    'turn': 2,
                    'question': 'When will it arrive?',
                    'intent': 'delivery_time',
                    'answer': 'It should arrive tomorrow.'
                }
            ]
        )
    ]
    
    # Validate data
    validation_report = processor.validate_conversation_data(sample_conversations)
    print(f"Validation report: {validation_report}")
    
    # Extract statistics
    stats = processor.extract_intent_statistics(sample_conversations)
    print(f"Statistics: {stats}")
    
    # Prepare HMM data
    hmm_data = processor.prepare_hmm_training_data(sample_conversations)
    print(f"HMM training data prepared: {list(hmm_data.keys())}")

if __name__ == "__main__":
    main()