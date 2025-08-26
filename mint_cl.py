import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle

@dataclass
class IntentHierarchy:
    """Data structure for hierarchical intent taxonomy"""
    level1_to_idx: Dict[str, int]
    level2_to_idx: Dict[str, int]
    level3_to_idx: Dict[str, int]
    idx_to_level1: Dict[int, str]
    idx_to_level2: Dict[int, str]
    idx_to_level3: Dict[int, str]
    hierarchy_map: Dict[str, Tuple[str, str, str]]  # intent -> (l1, l2, l3)

class MultiTurnDataset(Dataset):
    """Dataset for multi-turn intent classification with contrastive learning"""
    
    def __init__(self, conversations: List[Dict], tokenizer, hierarchy: IntentHierarchy, 
                 max_length: int = 512, include_responses: bool = True):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.hierarchy = hierarchy
        self.max_length = max_length
        self.include_responses = include_responses
        
        # Prepare data samples
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare training samples from conversations"""
        for conv in self.conversations:
            dialogue = conv.get('dialogue', [])
            if not dialogue:
                continue
                
            # Build conversation context for each turn
            for i, turn in enumerate(dialogue):
                # Collect all previous user utterances
                user_utterances = []
                for j in range(i + 1):
                    user_utterances.append(dialogue[j]['question'])
                
                # Create input sequence
                q_all = ", ".join(user_utterances)
                
                # Get intent labels for current turn
                intent = turn['intent']
                if intent not in self.hierarchy.hierarchy_map:
                    continue
                
                l1, l2, l3 = self.hierarchy.hierarchy_map[intent]
                
                sample = {
                    'input_text': q_all,
                    'intent': intent,
                    'level1_label': self.hierarchy.level1_to_idx[l1],
                    'level2_label': self.hierarchy.level2_to_idx[l2],
                    'level3_label': self.hierarchy.level3_to_idx[l3],
                    'turn_index': i,
                    'conversation_id': conv.get('conv_id', f"conv_{len(self.samples)}")
                }
                
                # Add response ranking data if available
                if self.include_responses:
                    answer = turn.get('answer', '')
                    alt_answer = turn.get('alternative_answer', '')
                    
                    if answer and alt_answer:
                        # Create positive and negative response pairs
                        sample['positive_response'] = answer
                        sample['negative_response'] = alt_answer
                        sample['has_contrastive'] = True
                    else:
                        sample['has_contrastive'] = False
                else:
                    sample['has_contrastive'] = False
                
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input text
        encoding = self.tokenizer(
            sample['input_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'level1_label': torch.tensor(sample['level1_label'], dtype=torch.long),
            'level2_label': torch.tensor(sample['level2_label'], dtype=torch.long),
            'level3_label': torch.tensor(sample['level3_label'], dtype=torch.long),
        }
        
        # Add contrastive learning data if available
        if sample['has_contrastive']:
            # Tokenize positive response pair
            pos_text = sample['input_text'] + " [SEP] " + sample['positive_response']
            pos_encoding = self.tokenizer(
                pos_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Tokenize negative response pair
            neg_text = sample['input_text'] + " [SEP] " + sample['negative_response']
            neg_encoding = self.tokenizer(
                neg_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            result.update({
                'has_contrastive': True,
                'pos_input_ids': pos_encoding['input_ids'].squeeze(),
                'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
                'neg_input_ids': neg_encoding['input_ids'].squeeze(),
                'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
            })
        else:
            result['has_contrastive'] = False
        
        return result

class HierarchicalTextClassifier(nn.Module):
    """Hierarchical text classifier with label attention mechanism"""
    
    def __init__(self, hierarchy: IntentHierarchy, hidden_dim: int = 768):
        super().__init__()
        self.hierarchy = hierarchy
        self.hidden_dim = hidden_dim
        
        # XLM-RoBERTa encoder
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        # Label attention layers for each level
        self.level1_projection = nn.Linear(hidden_dim, hidden_dim)
        self.level2_projection = nn.Linear(hidden_dim * 2, hidden_dim)  # Concatenated with level1
        self.level3_projection = nn.Linear(hidden_dim * 2, hidden_dim)  # Concatenated with level2
        
        # Classification heads
        self.level1_classifier = nn.Linear(hidden_dim, len(hierarchy.level1_to_idx))
        self.level2_classifier = nn.Linear(hidden_dim, len(hierarchy.level2_to_idx))
        self.level3_classifier = nn.Linear(hidden_dim, len(hierarchy.level3_to_idx))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        # Get contextualized embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        hidden_state = self.dropout(hidden_state)
        
        # Level 1 processing
        l1_repr = self.level1_projection(hidden_state)
        l1_logits = self.level1_classifier(l1_repr)
        
        # Level 2 processing (concatenate with level 1)
        l2_input = torch.cat([hidden_state, l1_repr], dim=1)
        l2_repr = self.level2_projection(l2_input)
        l2_logits = self.level2_classifier(l2_repr)
        
        # Level 3 processing (concatenate with level 2)
        l3_input = torch.cat([hidden_state, l2_repr], dim=1)
        l3_repr = self.level3_projection(l3_input)
        l3_logits = self.level3_classifier(l3_repr)
        
        return {
            'level1_logits': l1_logits,
            'level2_logits': l2_logits,
            'level3_logits': l3_logits,
            'final_hidden': hidden_state
        }

class MINTContrastiveClassifier(nn.Module):
    """
    MINT-CL: Multi-task contrastive learning framework for multi-turn intent classification.
    
    This model combines hierarchical intent classification with response ranking
    using contrastive learning to learn more robust representations.
    """
    
    def __init__(self, hierarchy: IntentHierarchy, hidden_dim: int = 768):
        super().__init__()
        self.hierarchy = hierarchy
        self.hidden_dim = hidden_dim
        
        # Main hierarchical classifier
        self.intent_classifier = HierarchicalTextClassifier(hierarchy, hidden_dim)
        
        # Response ranking head
        self.response_ranker = nn.Linear(hidden_dim, 1)
        
        # Loss weights
        self.intent_weight = 1.0
        self.contrastive_weight = 0.3
        
    def forward(self, batch):
        # Intent classification forward pass
        intent_outputs = self.intent_classifier(
            batch['input_ids'], 
            batch['attention_mask']
        )
        
        results = {
            'intent_outputs': intent_outputs
        }
        
        # Contrastive learning forward pass (if data available)
        has_contrastive_data = batch.get('has_contrastive', False)
        if torch.is_tensor(has_contrastive_data):
            has_contrastive_data = has_contrastive_data.any().item()
        
        if has_contrastive_data:
            # Process positive responses
            pos_outputs = self.intent_classifier(
                batch['pos_input_ids'],
                batch['pos_attention_mask']
            )
            pos_score = self.response_ranker(pos_outputs['final_hidden'])
            
            # Process negative responses
            neg_outputs = self.intent_classifier(
                batch['neg_input_ids'],
                batch['neg_attention_mask']
            )
            neg_score = self.response_ranker(neg_outputs['final_hidden'])
            
            results.update({
                'positive_score': pos_score,
                'negative_score': neg_score,
                'has_contrastive_data': True
            })
        else:
            results['has_contrastive_data'] = False
        
        return results
    
    def compute_loss(self, batch, outputs):
        """Compute combined loss for intent classification and contrastive learning"""
        losses = {}
        
        # Intent classification losses
        intent_outputs = outputs['intent_outputs']
        
        l1_loss = F.cross_entropy(
            intent_outputs['level1_logits'], 
            batch['level1_label']
        )
        l2_loss = F.cross_entropy(
            intent_outputs['level2_logits'], 
            batch['level2_label']
        )
        l3_loss = F.cross_entropy(
            intent_outputs['level3_logits'], 
            batch['level3_label']
        )
        
        intent_loss = (l1_loss + l2_loss + l3_loss) / 3
        losses['intent_loss'] = intent_loss
        
        # Contrastive loss (if applicable)
        contrastive_loss = 0
        if outputs.get('has_contrastive_data', False):
            pos_scores = outputs['positive_score']
            neg_scores = outputs['negative_score']
            
            # Contrastive loss: encourage higher scores for positive responses
            contrastive_loss = -torch.log(
                torch.sigmoid(pos_scores - neg_scores) + 1e-8
            ).mean()
            
            losses['contrastive_loss'] = contrastive_loss
        
        # Combined loss
        total_loss = (self.intent_weight * intent_loss + 
                     self.contrastive_weight * contrastive_loss)
        losses['total_loss'] = total_loss
        
        return losses
    
    def predict(self, batch):
        """Make predictions for intent classification"""
        with torch.no_grad():
            outputs = self.forward(batch)
            intent_outputs = outputs['intent_outputs']
            
            # Get predictions for each level
            l1_preds = torch.argmax(intent_outputs['level1_logits'], dim=1)
            l2_preds = torch.argmax(intent_outputs['level2_logits'], dim=1)
            l3_preds = torch.argmax(intent_outputs['level3_logits'], dim=1)
            
            return {
                'level1_predictions': l1_preds,
                'level2_predictions': l2_preds,
                'level3_predictions': l3_preds,
            }

class MINTTrainer:
    """Trainer class for MINT-CL model"""
    
    def __init__(self, model: MINTContrastiveClassifier, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_intent_loss = 0
        total_contrastive_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            losses = self.model.compute_loss(batch, outputs)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Track losses
            total_loss += losses['total_loss'].item()
            total_intent_loss += losses['intent_loss'].item()
            if 'contrastive_loss' in losses:
                total_contrastive_loss += losses['contrastive_loss'].item()
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_intent_loss = total_intent_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'intent_loss': avg_intent_loss,
            'contrastive_loss': avg_contrastive_loss
        }
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate the model"""
        self.model.eval()
        all_l1_preds, all_l1_true = [], []
        all_l2_preds, all_l2_true = [], []
        all_l3_preds, all_l3_true = [], []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                predictions = self.model.predict(batch)
                
                # Collect predictions and labels
                all_l1_preds.extend(predictions['level1_predictions'].cpu().numpy())
                all_l2_preds.extend(predictions['level2_predictions'].cpu().numpy())
                all_l3_preds.extend(predictions['level3_predictions'].cpu().numpy())
                
                all_l1_true.extend(batch['level1_label'].cpu().numpy())
                all_l2_true.extend(batch['level2_label'].cpu().numpy())
                all_l3_true.extend(batch['level3_label'].cpu().numpy())
                
                # Calculate loss
                outputs = self.model(batch)
                losses = self.model.compute_loss(batch, outputs)
                total_loss += losses['total_loss'].item()
                num_batches += 1
        
        # Calculate metrics
        l1_acc = accuracy_score(all_l1_true, all_l1_preds)
        l2_acc = accuracy_score(all_l2_true, all_l2_preds)
        l3_acc = accuracy_score(all_l3_true, all_l3_preds)
        
        l1_f1 = f1_score(all_l1_true, all_l1_preds, average='weighted')
        l2_f1 = f1_score(all_l2_true, all_l2_preds, average='weighted')
        l3_f1 = f1_score(all_l3_true, all_l3_preds, average='weighted')
        
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'level1_accuracy': l1_acc,
            'level2_accuracy': l2_acc,
            'level3_accuracy': l3_acc,
            'level1_f1': l1_f1,
            'level2_f1': l2_f1,
            'level3_f1': l3_f1,
            'overall_accuracy': (l1_acc + l2_acc + l3_acc) / 3
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              epochs: int = 5, learning_rate: float = 2e-5):
        """Full training loop"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0
        training_history = []
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, optimizer)
            self.logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"Intent Loss: {train_metrics['intent_loss']:.4f}, "
                           f"Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Overall Acc: {val_metrics['overall_accuracy']:.4f}")
            self.logger.info(f"L1 Acc: {val_metrics['level1_accuracy']:.4f}, "
                           f"L2 Acc: {val_metrics['level2_accuracy']:.4f}, "
                           f"L3 Acc: {val_metrics['level3_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['overall_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['overall_accuracy']
                self.save_model('best_model.pt')
                self.logger.info(f"New best model saved! Accuracy: {best_val_acc:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
        
        return training_history
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hierarchy': self.model.hierarchy,
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('hierarchy')

def build_intent_hierarchy(conversations: List[Dict]) -> IntentHierarchy:
    """
    Build intent hierarchy from conversation data.
    
    This is a simplified implementation. In practice, you would have
    predefined taxonomy or extract it from your specific domain data.
    """
    # Extract all intents
    all_intents = set()
    for conv in conversations:
        for turn in conv.get('dialogue', []):
            all_intents.add(turn['intent'])
    
    # Create simple 3-level hierarchy (this would be domain-specific)
    level1_intents = set()
    level2_intents = set()
    level3_intents = set()
    hierarchy_map = {}
    
    for intent in all_intents:
        # Simple mapping - in practice this would be domain-specific
        parts = intent.split('_')
        if len(parts) >= 1:
            l1 = parts[0]
            l2 = parts[0] + '_' + (parts[1] if len(parts) > 1 else 'general')
            l3 = intent
        else:
            l1 = 'general'
            l2 = 'general_misc'
            l3 = intent
        
        level1_intents.add(l1)
        level2_intents.add(l2)
        level3_intents.add(l3)
        hierarchy_map[intent] = (l1, l2, l3)
    
    # Create mappings
    level1_to_idx = {intent: i for i, intent in enumerate(sorted(level1_intents))}
    level2_to_idx = {intent: i for i, intent in enumerate(sorted(level2_intents))}
    level3_to_idx = {intent: i for i, intent in enumerate(sorted(level3_intents))}
    
    idx_to_level1 = {i: intent for intent, i in level1_to_idx.items()}
    idx_to_level2 = {i: intent for intent, i in level2_to_idx.items()}
    idx_to_level3 = {i: intent for intent, i in level3_to_idx.items()}
    
    return IntentHierarchy(
        level1_to_idx=level1_to_idx,
        level2_to_idx=level2_to_idx,
        level3_to_idx=level3_to_idx,
        idx_to_level1=idx_to_level1,
        idx_to_level2=idx_to_level2,
        idx_to_level3=idx_to_level3,
        hierarchy_map=hierarchy_map
    )

def main():
    """Example usage of MINT-CL"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load conversations (placeholder - would load from actual data)
    conversations = [
        {
            'conv_id': 'conv1',
            'dialogue': [
                {
                    'question': 'Where is my order?',
                    'intent': 'track_order',
                    'answer': 'Let me check your order status.',
                    'alternative_answer': 'I will look into that.'
                },
                {
                    'question': 'When will it arrive?', 
                    'intent': 'delivery_time',
                    'answer': 'It should arrive tomorrow.',
                    'alternative_answer': 'Expected delivery is soon.'
                }
            ]
        }
        # More conversations would be loaded here
    ]
    
    # Build intent hierarchy
    hierarchy = build_intent_hierarchy(conversations)
    print(f"Built hierarchy with {len(hierarchy.level1_to_idx)} L1, "
          f"{len(hierarchy.level2_to_idx)} L2, {len(hierarchy.level3_to_idx)} L3 intents")
    
    # Initialize tokenizer and datasets
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    train_dataset = MultiTurnDataset(conversations, tokenizer, hierarchy)
    val_dataset = MultiTurnDataset(conversations, tokenizer, hierarchy)  # Same data for demo
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model and trainer
    model = MINTContrastiveClassifier(hierarchy)
    trainer = MINTTrainer(model, tokenizer, device='cpu')  # Use CPU for demo
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_dataloader, val_dataloader, epochs=2, learning_rate=2e-5)
    
    print("Training completed!")
    
    # Save final results
    with open('training_history.json', 'w') as f:
        # Convert tensors to floats for JSON serialization
        serializable_history = []
        for epoch_data in history:
            epoch_dict = {
                'epoch': epoch_data['epoch'],
                'train_metrics': {k: float(v) if torch.is_tensor(v) else v 
                                for k, v in epoch_data['train_metrics'].items()},
                'val_metrics': {k: float(v) if torch.is_tensor(v) else v 
                              for k, v in epoch_data['val_metrics'].items()}
            }
            serializable_history.append(epoch_dict)
        
        json.dump(serializable_history, f, indent=2)
    
    print("Results saved!")

if __name__ == "__main__":
    main()