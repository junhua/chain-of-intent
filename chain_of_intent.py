import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
from collections import Counter, defaultdict
import json
import logging
from dataclasses import dataclass
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class IntentTransition:
    """Data class for storing intent transition probabilities"""
    initial_distribution: Dict[str, float]
    transition_matrix: Dict[str, Dict[str, float]]
    turn_distribution: Dict[int, float]

class ChainOfIntent:
    """
    Chain-of-Intent implementation using HMM enhanced with LLMs for dialogue generation.
    
    This class implements the core Chain-of-Intent mechanism described in the paper,
    which combines Hidden Markov Models with Large Language Models to generate
    contextually aware, intent-driven conversations through self-play.
    """
    
    def __init__(self, intent_data_path: str = None, llm_model: str = "gpt-3.5-turbo"):
        self.intent_transitions = None
        self.intent_to_questions = defaultdict(list)
        self.llm_model = llm_model
        self.logger = logging.getLogger(__name__)
        
        # Load intent data if provided
        if intent_data_path:
            self.load_intent_data(intent_data_path)
    
    def extract_domain_knowledge(self, chat_logs: List[Dict]) -> IntentTransition:
        """
        Extract domain knowledge from historical chat logs.
        
        Args:
            chat_logs: List of chat log dictionaries with keys:
                - 'conversation_id': str
                - 'turn': int 
                - 'question': str
                - 'intent': str
                - 'answer': str
        
        Returns:
            IntentTransition object containing extracted knowledge
        """
        self.logger.info("Extracting domain knowledge from chat logs...")
        
        # Group conversations by ID
        conversations = defaultdict(list)
        for log in chat_logs:
            conversations[log['conversation_id']].append(log)
        
        # Sort each conversation by turn
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda x: x['turn'])
        
        # Extract turn distribution P(T)
        turn_counts = Counter([len(conv) for conv in conversations.values()])
        total_conversations = len(conversations)
        turn_distribution = {turns: count/total_conversations 
                           for turns, count in turn_counts.items()}
        
        # Extract initial intent distribution P(I_1)
        initial_intents = [conv[0]['intent'] for conv in conversations.values()]
        initial_intent_counts = Counter(initial_intents)
        initial_distribution = {intent: count/len(initial_intents) 
                              for intent, count in initial_intent_counts.items()}
        
        # Extract intent transition matrix P(I_t|I_{t-1})
        transition_counts = defaultdict(Counter)
        for conv in conversations.values():
            for i in range(1, len(conv)):
                prev_intent = conv[i-1]['intent']
                curr_intent = conv[i]['intent']
                transition_counts[prev_intent][curr_intent] += 1
        
        # Normalize transition probabilities
        transition_matrix = {}
        for prev_intent, next_intents in transition_counts.items():
            total = sum(next_intents.values())
            transition_matrix[prev_intent] = {
                intent: count/total for intent, count in next_intents.items()
            }
        
        # Build intent-to-questions mapping
        for log in chat_logs:
            self.intent_to_questions[log['intent']].append(log['question'])
        
        self.intent_transitions = IntentTransition(
            initial_distribution=initial_distribution,
            transition_matrix=transition_matrix,
            turn_distribution=turn_distribution
        )
        
        self.logger.info(f"Extracted knowledge: {len(initial_distribution)} intents, "
                        f"{len(turn_distribution)} turn lengths")
        
        return self.intent_transitions
    
    def sample_intent_sequence(self) -> Tuple[int, List[str]]:
        """
        Sample a sequence of intents using the HMM.
        
        Returns:
            Tuple of (number_of_turns, list_of_intents)
        """
        if not self.intent_transitions:
            raise ValueError("Intent transitions not loaded. Call extract_domain_knowledge first.")
        
        # Sample number of turns
        turns = np.random.choice(
            list(self.intent_transitions.turn_distribution.keys()),
            p=list(self.intent_transitions.turn_distribution.values())
        )
        
        # Sample initial intent
        intents = [np.random.choice(
            list(self.intent_transitions.initial_distribution.keys()),
            p=list(self.intent_transitions.initial_distribution.values())
        )]
        
        # Sample subsequent intents based on transition probabilities
        for t in range(1, turns):
            prev_intent = intents[t-1]
            if prev_intent in self.intent_transitions.transition_matrix:
                next_intents = list(self.intent_transitions.transition_matrix[prev_intent].keys())
                next_probs = list(self.intent_transitions.transition_matrix[prev_intent].values())
                next_intent = np.random.choice(next_intents, p=next_probs)
            else:
                # Fallback to uniform sampling if transition not found
                next_intent = random.choice(list(self.intent_transitions.initial_distribution.keys()))
            intents.append(next_intent)
        
        return turns, intents
    
    def sample_reference_question(self, intent: str) -> str:
        """
        Sample a reference question X_t for the given intent.
        
        Args:
            intent: The intent to sample a question for
            
        Returns:
            A sample question string
        """
        if intent not in self.intent_to_questions:
            return f"Sample question for {intent}"
        
        return random.choice(self.intent_to_questions[intent])
    
    def generate_question_with_llm(self, intent: str, reference_question: str, 
                                 conversation_history: List[Tuple[str, str]]) -> str:
        """
        Generate a contextually aware question using LLM.
        
        Args:
            intent: The target intent for the question
            reference_question: A sample question for the intent
            conversation_history: List of (question, answer) pairs
            
        Returns:
            Generated question string
        """
        # Build conversation context
        context = ""
        for i, (q, a) in enumerate(conversation_history):
            context += f"Turn {i+1}:\nUser: {q}\nAgent: {a}\n\n"
        
        # Create instruction prompt
        instruction = f"""You are helping generate a realistic e-commerce customer service conversation.

Current Intent: {intent}
Reference Question: {reference_question}

Conversation History:
{context}

Generate the next user question that:
1. Has the intent: {intent}
2. Is contextually coherent with the conversation history
3. Uses natural language with appropriate pronouns and references
4. Avoids unnecessary repetition of previous context
5. Sounds like a real customer inquiry

Generate only the user's question (no additional text or formatting):"""

        try:
            if "gpt" in self.llm_model.lower():
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": instruction}],
                    max_tokens=100,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            else:
                # Use local model (placeholder implementation)
                return self._generate_with_local_model(instruction)
                
        except Exception as e:
            self.logger.error(f"Error generating question with LLM: {e}")
            # Fallback to reference question with minor modification
            return f"{reference_question}?"
    
    def generate_answer_with_llm(self, question: str, 
                               conversation_history: List[Tuple[str, str]]) -> str:
        """
        Generate an appropriate answer using LLM.
        
        Args:
            question: The current user question
            conversation_history: List of (question, answer) pairs
            
        Returns:
            Generated answer string
        """
        # Build conversation context
        context = ""
        for i, (q, a) in enumerate(conversation_history):
            context += f"Turn {i+1}:\nUser: {q}\nAgent: {a}\n\n"
        
        instruction = f"""You are an e-commerce customer service agent. Provide a helpful, concise response.

Conversation History:
{context}

Current User Question: {question}

Generate a helpful customer service response that:
1. Directly addresses the user's question
2. Is professional and friendly
3. Is contextually appropriate
4. Provides specific information when possible

Generate only the agent's response (no additional text or formatting):"""

        try:
            if "gpt" in self.llm_model.lower():
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": instruction}],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            else:
                return self._generate_with_local_model(instruction)
                
        except Exception as e:
            self.logger.error(f"Error generating answer with LLM: {e}")
            return "I understand your concern. Let me help you with that."
    
    def _generate_with_local_model(self, prompt: str) -> str:
        """Placeholder for local model generation"""
        # This would be implemented with a local model like Llama
        return "Generated response from local model"
    
    def generate_conversation(self) -> Dict:
        """
        Generate a complete conversation using Chain-of-Intent.
        
        Returns:
            Dictionary containing the generated conversation
        """
        # Sample intent sequence
        turns, intents = self.sample_intent_sequence()
        
        conversation = {
            'turns': turns,
            'intents': intents,
            'dialogue': []
        }
        
        conversation_history = []
        
        for t in range(turns):
            intent = intents[t]
            
            # Sample reference question
            reference_question = self.sample_reference_question(intent)
            
            # Generate contextual question
            question = self.generate_question_with_llm(
                intent, reference_question, conversation_history
            )
            
            # Generate answer
            answer = self.generate_answer_with_llm(question, conversation_history)
            
            # Add to conversation
            conversation['dialogue'].append({
                'turn': t + 1,
                'intent': intent,
                'question': question,
                'answer': answer,
                'reference_question': reference_question
            })
            
            # Update history
            conversation_history.append((question, answer))
        
        return conversation
    
    def generate_alternative_answer(self, conversation: Dict, 
                                  alternative_model: str = "llama-3-8b") -> Dict:
        """
        Generate an alternative answer for the last turn using a different model.
        
        Args:
            conversation: The original conversation
            alternative_model: Model to use for alternative generation
            
        Returns:
            Modified conversation with alternative answer
        """
        if not conversation['dialogue']:
            return conversation
        
        last_turn = conversation['dialogue'][-1]
        history = [(turn['question'], turn['answer']) 
                  for turn in conversation['dialogue'][:-1]]
        
        # Generate alternative answer (simplified implementation)
        alt_answer = self.generate_answer_with_llm(
            last_turn['question'], history
        )
        
        # Create alternative conversation
        alt_conversation = conversation.copy()
        alt_conversation['dialogue'] = conversation['dialogue'].copy()
        alt_conversation['dialogue'][-1] = last_turn.copy()
        alt_conversation['dialogue'][-1]['alternative_answer'] = alt_answer
        
        return alt_conversation
    
    def evaluate_answer_quality(self, conversation_history: List[Tuple[str, str]], 
                               question: str, answer: str) -> float:
        """
        Evaluate answer quality using GPT-4 (placeholder implementation).
        
        Args:
            conversation_history: Previous turns
            question: Current question
            answer: Answer to evaluate
            
        Returns:
            Quality score (1-10)
        """
        # Build context
        context = ""
        for i, (q, a) in enumerate(conversation_history):
            context += f"Turn {i+1}:\nUser: {q}\nAgent: {a}\n\n"
        
        evaluation_prompt = f"""Rate the quality of this customer service response on a scale of 1-10.

Conversation History:
{context}

Current Question: {question}
Response to Evaluate: {answer}

Consider:
- Relevance to the question
- Helpfulness and completeness
- Professional tone
- Contextual appropriateness

Provide only a numeric score (1-10):"""

        try:
            if "gpt-4" in self.llm_model:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                score_text = response.choices[0].message.content.strip()
                return float(score_text)
            else:
                # Placeholder scoring
                return random.uniform(6.0, 9.0)
                
        except Exception as e:
            self.logger.error(f"Error evaluating answer quality: {e}")
            return 7.0
    
    def save_conversations(self, conversations: List[Dict], filepath: str):
        """Save generated conversations to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    def load_intent_data(self, filepath: str):
        """Load pre-extracted intent transition data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.intent_transitions = IntentTransition(
            initial_distribution=data['initial_distribution'],
            transition_matrix=data['transition_matrix'],
            turn_distribution=data['turn_distribution']
        )
        
        if 'intent_to_questions' in data:
            self.intent_to_questions = defaultdict(list, data['intent_to_questions'])

def main():
    """Example usage of Chain-of-Intent"""
    # Initialize Chain-of-Intent
    coi = ChainOfIntent(llm_model="gpt-3.5-turbo")
    
    # Example chat logs (in practice, this would be loaded from a dataset)
    example_chat_logs = [
        {
            'conversation_id': 'conv1',
            'turn': 1,
            'question': 'Where is my order?',
            'intent': 'track_order',
            'answer': 'Let me check your order status.'
        },
        {
            'conversation_id': 'conv1', 
            'turn': 2,
            'question': 'When will it arrive?',
            'intent': 'delivery_time',
            'answer': 'Your order will arrive tomorrow.'
        }
        # More examples would be added here
    ]
    
    # Extract domain knowledge
    coi.extract_domain_knowledge(example_chat_logs)
    
    # Generate conversations
    conversations = []
    for i in range(10):
        conv = coi.generate_conversation()
        conversations.append(conv)
        print(f"Generated conversation {i+1} with {conv['turns']} turns")
    
    # Save conversations
    coi.save_conversations(conversations, 'generated_conversations.json')
    print("Conversations saved successfully!")

if __name__ == "__main__":
    main()