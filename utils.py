"""
Utility functions and classes for GRPO fine-tuning lab.
This module contains functions and classes that were extracted from the notebook
to reduce its length and improve code organization.
"""

import os
import re
import random
import logging
import torch
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, TrainerCallback
from datasets import load_from_disk


def setup_logging(log_dir: str = "./grpo_logs", device=None):
    """
    Setup logging directory and configuration.
    
    Args:
        log_dir: Directory to store log files
        device: Device information to log
    
    Returns:
        logger: Configured logger instance
        log_file: Path to the log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging - write ONLY to file, no console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file)  # Only log to file
        ],
        force=True  # Override any existing logging configuration
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("GRPO Training Lab Started")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    if device:
        logger.info(f"Device: {device}")

    print(f"✅ Logging configured - all detailed logs will be written to: {log_file}")
    print("Note: Detailed reward computation logs will only appear in the log file, not in console output")
    
    return logger, log_file


def load_and_explore_gsm8k_dataset():
    """
    Load GSM8K dataset and display basic information.
    
    Returns:
        dataset: The loaded GSM8K dataset
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading GSM8K dataset...")
    dataset = load_from_disk("/app/data/gsm8k")

    # Display dataset information
    print("Dataset Structure:")
    print(dataset)
    print("\nDataset splits:")
    print(f"- Train: {len(dataset['train'])} examples")
    print(f"- Test: {len(dataset['test'])} examples")

    # Show a sample problem
    print("\n" + "="*60)
    print("Sample Problem from Training Set:")
    print("="*60)
    example = dataset["train"][0]
    print(f"\n📝 Question:\n{example['question']}")
    print(f"\n✅ Answer:\n{example['answer']}")

    # Extract the numerical answer
    if "####" in example['answer']:
        numerical_answer = example['answer'].split("####")[-1].strip()
        print(f"\n🔢 Numerical Answer: {numerical_answer}")
    
    return dataset


def truncate_prompt(text: str, tokenizer, max_length: int) -> str:
    """
    Truncate prompt to fit within token limit.
    We truncate from the left to preserve the end of the question.
    """
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(tokens) <= max_length:
        return text
    # Truncate from left to preserve end context
    truncated_tokens = tokens[-max_length:]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def prepare_dataset(config, tokenizer):
    """
    Prepare GSM8K dataset for GRPO training.
    
    Steps:
    1. Load the dataset
    2. Split into train/validation
    3. Format prompts
    4. Extract numerical answers
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing dataset for training...")
    
    # Load dataset
    from datasets.utils import disable_progress_bar
    disable_progress_bar()
    dataset = load_from_disk(
        "/app/data/gsm8k")
    
    # Use the TRAIN set for training (not test set!)
    train_data_full = dataset["train"]
    logger.info(f"GSM8K train set size: {len(train_data_full)}")
    
    # Split train set into train and validation
    total_size = len(train_data_full)
    train_size = int(total_size * config.train_split_ratio)
    
    # Create consistent split using seed
    indices = list(range(total_size))
    random.Random(config.seed).shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    
    train_data = train_data_full.select(train_indices)
    eval_data = train_data_full.select(eval_indices)
    
    logger.info(f"Dataset sizes - Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Process data into required format
    def process_example(example):
        question = example['question']
        answer_text = example['answer']
        
        # Extract numerical answer
        if "####" in answer_text:
            answer = answer_text.split("####")[-1].strip()
            answer = answer.replace(',', '').replace('$', '')
            try:
                answer_num = float(answer)
            except:
                answer_num = 0.0
        else:
            answer_num = 0.0
        
        # Create prompt that encourages step-by-step solutions
        prompt = f"""Question: {question}

Let's solve this step-by-step and find the numerical answer:
"""
        
        # Truncate prompt if too long
        prompt = truncate_prompt(prompt, tokenizer, config.max_prompt_length)
        
        return {
            'prompt': prompt,
            'question': question,
            'answer': answer_num,
            'answer_text': answer_text
        }
    
    # Apply processing
    train_dataset = train_data.map(process_example)
    eval_dataset = eval_data.map(process_example)
    
    return train_dataset, eval_dataset


class GSM8KEvaluationCallback(TrainerCallback):
    """
    Custom callback to evaluate on GSM8K test set during training.
    Uses full test set (1319 examples) for most accurate evaluation.
    """
    
    def __init__(self, tokenizer, test_dataset, batch_size=64, sample_size=1.0):
        # Use sample_size=1.0 to evaluate on full test set
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.sample_size = sample_size  # 1.0 = use all test examples
        self.logger = logging.getLogger(__name__)
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step - run evaluation here"""
        if state.global_step > 0 and state.global_step % args.eval_steps == 0:
            self.logger.info(f"🚀 EVALUATION TRIGGER at step {state.global_step}")
            self._run_evaluation(state, kwargs.get('model', model))
            
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Called after evaluation phase - run our custom test evaluation"""
        self.logger.info(f"🔥 EVALUATION CALLBACK TRIGGERED at step {state.global_step}")
        self._run_evaluation(state, kwargs.get('model', model))
    
    def _run_evaluation(self, state, model):
        """Run the actual evaluation on full test set"""
        self.logger.info(f"🔍 Starting evaluation at step {state.global_step}")
        
        if model is None:
            self.logger.error("❌ Model is None - cannot evaluate")
            return
            
        self.logger.info("="*80)
        self.logger.info("GSM8K TEST SET EVALUATION")
        self.logger.info("="*80)
        self.logger.info(f"Using batch_size={self.batch_size}, sample_size={self.sample_size} (FULL TEST SET)")
        
        try:
            # Ensure model is in eval mode
            model.eval()
            
            # Run evaluation on test set
            with torch.no_grad():
                results = self.evaluate_model(
                    model, 
                    self.tokenizer, 
                    self.test_dataset, 
                    sample_size=self.sample_size, 
                    batch_size=self.batch_size
                )
            
            # Log results
            self.logger.info(f"Step: {state.global_step}")
            self.logger.info(f"Epoch: {state.epoch if state.epoch else 0}")
            self.logger.info(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            
            # Also print to console
            print(f"\n📊 Step {state.global_step} - Test Accuracy: {results['accuracy']*100:.2f}% ({results['correct_count']}/{results['total_examples']})")
            
            # Switch back to train mode
            model.train()
            
        except Exception as e:
            self.logger.error(f"Error during GSM8K evaluation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        self.logger.info("="*80)
    
    def extract_numerical_answer(self, answer_text: str) -> str:
        """Extract the final numerical answer from GSM8K answer text."""
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        else:
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            return numbers[-1] if numbers else "0"
    
    def extract_final_answer(self, response: str) -> str:
        """Extract the final numerical answer from model response."""
        # Look for patterns like "The answer is X" or numbers at the end
        patterns = [
            r"The answer is ([+-]?\d+\.?\d*)",
            r"= ([+-]?\d+\.?\d*)",
            r"([+-]?\d+\.?\d*)\s*$"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()
        
        # Fallback: extract last number
        numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
        return numbers[-1] if numbers else "0"
    
    def generate_batch_responses(self, model, tokenizer, questions: List[str], batch_size: int = 4) -> List[str]:
        """Generate model responses for multiple questions using batch processing."""
        responses = []
        
        # Process questions in batches
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating responses"):
            batch_questions = questions[i:i + batch_size]
            
            # Format prompts for math problems
            batch_prompts = [f"Problem: {question}\nSolution: Let me solve this step by step.\n" 
                            for question in batch_questions]
            
            # Tokenize
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode responses
            for j in range(len(batch_questions)):
                output_ids = outputs[j, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
                responses.append(response)
        
        return responses
    
    def evaluate_model(self, model, tokenizer, dataset, sample_size: float = 1.0, batch_size=4) -> Dict:
        """
        Evaluate model on dataset using full test set for maximum accuracy
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            dataset: The GSM8K dataset
            sample_size: Fraction of dataset to use (1.0 = full dataset)
            batch_size: Batch size for generation
        """
        total_size = len(dataset)
        
        if sample_size >= 1.0:
            # Use full dataset
            eval_dataset = dataset
            eval_size = total_size
            print(f"  📊 Evaluating on FULL test set: {eval_size} examples")
        else:
            # Sample the data if needed
            eval_size = int(total_size * sample_size)
            eval_indices = random.sample(range(total_size), eval_size)
            eval_dataset = dataset.select(eval_indices)
            print(f"  📊 Evaluating on {eval_size} examples ({sample_size*100:.0f}% of {total_size} test examples)")
        
        # Extract questions and correct answers
        questions = [example['question'] for example in eval_dataset]
        correct_answers = [self.extract_numerical_answer(example['answer']) for example in eval_dataset]
        
        # Generate responses using batch processing
        model_responses = self.generate_batch_responses(model, tokenizer, questions, batch_size=batch_size)
        
        # Calculate accuracy
        correct_count = 0
        results = {
            'total_examples': eval_size,
            'predictions': [],
            'correct': []
        }
        
        for i, (correct_answer, response) in enumerate(zip(correct_answers, model_responses)):
            predicted_answer = self.extract_final_answer(response)
            
            # Check if correct
            try:
                correct_num = float(correct_answer)
                predicted_num = float(predicted_answer)
                is_correct = abs(predicted_num - correct_num) < 1e-6
            except:
                is_correct = False
            
            if is_correct:
                correct_count += 1
                
            results['predictions'].append(predicted_answer)
            results['correct'].append(is_correct)
        
        results['accuracy'] = correct_count / eval_size
        results['correct_count'] = correct_count
        
        return results


def evaluate_and_compare(original_model, finetuned_model_path, tokenizer, test_dataset, reward_model, num_samples=50):
    """
    Compare original and fine-tuned models on test set.
    """
    print("\n📊 Model Comparison on Test Set")
    print("="*60)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Sample test set
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    sampled_data = test_dataset.select(indices)
    
    # reward_model is now passed as a parameter
    
    # Evaluate both models
    results = {'original': {'correct': 0, 'total_reward': 0},
               'finetuned': {'correct': 0, 'total_reward': 0}}
    
    for i, example in enumerate(tqdm(sampled_data, desc="Evaluating")):
        # Prepare prompt
        prompt = f"Question: {example['question']}\n\nLet's solve this step-by-step and find the numerical answer:\n"
        
        # Get correct answer
        correct_answer_text = example['answer']
        if "####" in correct_answer_text:
            correct_answer = float(correct_answer_text.split("####")[-1].strip().replace(',', '').replace('$', ''))
        else:
            correct_answer = 0.0
        
        # Evaluate both models
        for model_name, model in [('original', original_model), ('finetuned', finetuned_model)]:
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                predicted = reward_model.extract_numerical_answer(response)
                
                # Check if correct
                if predicted is not None and abs(predicted - correct_answer) < 0.01:
                    results[model_name]['correct'] += 1
                
                # Calculate reward
                reward = reward_model.compute_reward(response, correct_answer)
                results[model_name]['total_reward'] += reward
        
        # Show example comparisons for first few problems
        if i < 3:
            print(f"\n📝 Example {i+1}:")
            print(f"Question: {example['question'][:100]}...")
            print(f"Correct Answer: {correct_answer}")
    
    # Calculate metrics
    for model_name in ['original', 'finetuned']:
        results[model_name]['accuracy'] = results[model_name]['correct'] / num_samples
        results[model_name]['avg_reward'] = results[model_name]['total_reward'] / num_samples
    
    # Display results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print("\nOriginal Model:")
    print(f"  Accuracy: {results['original']['accuracy']*100:.2f}% ({results['original']['correct']}/{num_samples})")
    print(f"  Average Reward: {results['original']['avg_reward']:.3f}")
    
    print("\nFine-tuned Model:")
    print(f"  Accuracy: {results['finetuned']['accuracy']*100:.2f}% ({results['finetuned']['correct']}/{num_samples})")
    print(f"  Average Reward: {results['finetuned']['avg_reward']:.3f}")
    
    print("\n📈 Improvement:")
    acc_improvement = (results['finetuned']['accuracy'] - results['original']['accuracy']) * 100
    reward_improvement = results['finetuned']['avg_reward'] - results['original']['avg_reward']
    print(f"  Accuracy: {acc_improvement:+.2f} percentage points")
    print(f"  Average Reward: {reward_improvement:+.3f}")
    
    if acc_improvement > 0:
        print("\n🎉 The fine-tuned model shows improvement!")
    else:
        print("\n📝 Note: Training may need more epochs or data for better results.")
    
    return results
