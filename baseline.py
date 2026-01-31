"""
Baseline Evaluation: DeepSeek R1 on Towers of Hanoi

This script evaluates DeepSeek R1's performance on the Towers of Hanoi puzzle
across different problem difficulties using a local vllm server.
"""

import os
import json
import time
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from planning import TowersOfHanoiDataset, TowersOfHanoiValidator
from prompts import create_standard_prompt


# ============================================================================
# Configuration
# ============================================================================

# Problem distribution as specified
PROBLEM_DISTRIBUTION = [
    (3, 3, "3 disks"),
    (4, 3, "4 disks"),
    (5, 3, "5 disks"),
]

# Local vllm server configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASE_URL = "http://localhost:8000/v1"

# Output configuration
OUTPUT_DIR = "./baseline_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Client Setup
# ============================================================================

client = OpenAI(
    api_key="token-not-needed",
    base_url=BASE_URL,
)


# ============================================================================
# Prompt Template (imported from prompts.py)
# ============================================================================
# create_standard_prompt is now imported from prompts.py to avoid duplication

def create_prompt(problem: Dict) -> Tuple[str, str]:
    """
    Create the prompt for DeepSeek R1 based on the problem.
    
    Args:
        problem: Dictionary containing problem details from TowersOfHanoiDataset
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    num_disks = problem['num_disks']
    goal_peg = problem['goal_peg']
    
    # Use shared function from prompts.py
    return create_standard_prompt(num_disks, goal_peg)


# ============================================================================
# Evaluation Functions
# ============================================================================

def generate_solution(system_prompt: str, user_prompt: str) -> Dict:
    """
    Generate a solution using DeepSeek R1.
    
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        
    Returns:
        Dictionary containing reasoning, answer, and full output
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0,
            stream=False,
            max_completion_tokens=15_000,
            timeout=600.0,
        )
        
        # Check if response has reasoning_content attribute
        thinking = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        answer = response.choices[0].message.content
        
        # Combine for full output
        full_output = ""
        if thinking:
            full_output += f"<think>\n{thinking}\n</think>\n\n"
        full_output += answer
        
        return {
            'reasoning_content': thinking,
            'response_content': answer,
            'full_output': full_output,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'reasoning_content': "",
            'response_content': "",
            'full_output': "",
            'success': False,
            'error': str(e)
        }


def evaluate_single_problem(
    problem: Dict,
    validator: TowersOfHanoiValidator,
    trial_num: int,
    total_trials: int
) -> Dict:
    """
    Evaluate DeepSeek R1 on a single Towers of Hanoi problem.
    
    Args:
        problem: Problem dictionary from TowersOfHanoiDataset
        validator: TowersOfHanoiValidator instance
        trial_num: Current trial number
        total_trials: Total number of trials for this disk count
        
    Returns:
        Dictionary containing evaluation results
    """
    num_disks = problem['num_disks']
    goal_peg = problem['goal_peg']
    
    print(f"  Trial {trial_num}/{total_trials}: {num_disks} disks... ", end='', flush=True)
    
    # Create prompt
    system_prompt, user_prompt = create_prompt(problem)
    
    # Generate response
    start_time = time.time()
    result = generate_solution(system_prompt, user_prompt)
    generation_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ API Error: {result['error']}")
        return {
            'num_disks': num_disks,
            'trial_num': trial_num,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'full_output': "",
            'reasoning_content': "",
            'response_content': "",
            'success': False,
            'error': result['error'],
            'generation_time': generation_time,
            'reward': 0.0,
            'violations': -1,
            'correct': False,
        }
    
    # Validate the solution
    problem_state = {
        'num_disks': num_disks,
        'goal_peg': goal_peg
    }
    
    reward, violations = validator.validate_trace(result['full_output'], problem_state)
    
    # Determine correctness (reward of 1.0+ means goal reached)
    correct = reward >= 1.0
    
    # Print result
    status = "✓" if correct else "✗"
    print(f"{status} Reward: {reward:.2f}, Violations: {violations}, Time: {generation_time:.1f}s")
    
    return {
        'num_disks': num_disks,
        'trial_num': trial_num,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'full_output': result['full_output'],
        'reasoning_content': result['reasoning_content'],
        'response_content': result['response_content'],
        'success': True,
        'error': None,
        'generation_time': generation_time,
        'reward': reward,
        'violations': violations,
        'correct': correct,
        'optimal_moves': len(problem['optimal_moves']),
    }


def run_evaluation() -> Dict:
    """
    Run the full baseline evaluation across all problem difficulties.
    
    Returns:
        Dictionary containing all results and statistics
    """
    print("=" * 80)
    print("DeepSeek R1 Baseline Evaluation - Towers of Hanoi")
    print("=" * 80)
    print()
    
    # Initialize components
    dataset = TowersOfHanoiDataset(min_disks=3, max_disks=8)
    validator = TowersOfHanoiValidator()
    
    # Storage for all results
    all_results = []
    
    # Run evaluation for each problem configuration
    total_problems = sum(count for _, count, _ in PROBLEM_DISTRIBUTION)
    completed = 0
    
    for num_disks, num_trials, description in PROBLEM_DISTRIBUTION:
        print(f"\n{'='*80}")
        print(f"Testing {num_disks} Disks - {description}")
        print(f"{'='*80}")
        
        for trial in range(1, num_trials + 1):
            # Generate problem
            problem = dataset.generate_problem(num_disks=num_disks)
            
            # Evaluate
            result = evaluate_single_problem(
                problem, validator, trial, num_trials
            )
            all_results.append(result)
            
            completed += 1
        
        disk_results = [r for r in all_results if r['num_disks'] == num_disks]
        correct_count = sum(1 for r in disk_results if r['correct'])
        avg_reward = sum(r['reward'] for r in disk_results) / len(disk_results)
        avg_violations = sum(r['violations'] for r in disk_results if r['violations'] >= 0) / len(disk_results)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {num_disks} DISKS: {correct_count}/{num_trials} correct")
        print(f"{'='*80}")
        print(f"  Accuracy: {100*correct_count/num_trials:.1f}%")
        print(f"  Avg Reward: {avg_reward:.3f}")
        print(f"  Avg Violations: {avg_violations:.1f}")
        print(f"\nProgress: {completed}/{total_problems} problems completed")
    
    return {
        'timestamp': TIMESTAMP,
        'model': MODEL_NAME,
        'problem_distribution': PROBLEM_DISTRIBUTION,
        'results': all_results,
    }


def save_results(evaluation_data: Dict) -> None:
    """
    Save evaluation results to disk.
    
    Args:
        evaluation_data: Dictionary containing all evaluation results
    """
    # Create output directory - all files go in outputs_TIMESTAMP
    outputs_dir = os.path.join(OUTPUT_DIR, f"outputs_{TIMESTAMP}")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Save full results (including all outputs)
    full_results_path = os.path.join(outputs_dir, f"deepseek_r1_full_{TIMESTAMP}.json")
    with open(full_results_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    print(f"\n✓ Full results saved to: {full_results_path}")
    
    # Save summary statistics
    summary = compute_summary_statistics(evaluation_data)
    summary_path = os.path.join(outputs_dir, f"deepseek_r1_summary_{TIMESTAMP}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    for i, result in enumerate(evaluation_data['results']):
        if result['success']:
            output_path = os.path.join(
                outputs_dir, 
                f"output_{result['num_disks']}disks_trial{result['trial_num']:03d}.txt"
            )
            with open(output_path, 'w') as f:
                f.write(f"Problem: {result['num_disks']} disks\n")
                f.write(f"Trial: {result['trial_num']}\n")
                f.write(f"Correct: {result['correct']}\n")
                f.write(f"Reward: {result['reward']}\n")
                f.write(f"Violations: {result['violations']}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"SYSTEM PROMPT:\n{'='*80}\n\n")
                f.write(result['system_prompt'])
                f.write(f"\n\n{'='*80}\n")
                f.write(f"USER PROMPT:\n{'='*80}\n\n")
                f.write(result['user_prompt'])
                f.write(f"\n\n{'='*80}\n")
                f.write(f"FULL OUTPUT (WITH THINKING):\n{'='*80}\n\n")
                f.write(result['full_output'])
    
    print(f"✓ Individual outputs saved to: {outputs_dir}/")


def compute_summary_statistics(evaluation_data: Dict) -> Dict:
    """
    Compute summary statistics from evaluation results.
    
    Args:
        evaluation_data: Full evaluation data
        
    Returns:
        Dictionary containing summary statistics
    """
    results = evaluation_data['results']
    
    # Overall statistics
    total = len(results)
    successful_api_calls = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r['correct'])
    avg_reward = sum(r['reward'] for r in results) / total
    avg_violations = sum(r['violations'] for r in results if r['violations'] >= 0) / successful_api_calls
    avg_time = sum(r['generation_time'] for r in results) / total
    
    # Per-disk statistics
    per_disk_stats = {}
    for num_disks, num_trials, description in evaluation_data['problem_distribution']:
        disk_results = [r for r in results if r['num_disks'] == num_disks]
        if disk_results:
            correct_count = sum(1 for r in disk_results if r['correct'])
            per_disk_stats[num_disks] = {
                'description': description,
                'num_trials': num_trials,
                'correct': correct_count,
                'accuracy': correct_count / num_trials,
                'avg_reward': sum(r['reward'] for r in disk_results) / len(disk_results),
                'avg_violations': sum(r['violations'] for r in disk_results if r['violations'] >= 0) / len(disk_results),
                'avg_time': sum(r['generation_time'] for r in disk_results) / len(disk_results),
            }
    
    return {
        'timestamp': evaluation_data['timestamp'],
        'model': evaluation_data['model'],
        'overall': {
            'total_problems': total,
            'successful_api_calls': successful_api_calls,
            'correct_solutions': correct,
            'overall_accuracy': correct / total,
            'avg_reward': avg_reward,
            'avg_violations': avg_violations,
            'avg_generation_time': avg_time,
        },
        'per_disk_count': per_disk_stats,
    }


def print_final_summary(evaluation_data: Dict) -> None:
    """
    Print a formatted summary of the evaluation results.
    
    Args:
        evaluation_data: Full evaluation data
    """
    summary = compute_summary_statistics(evaluation_data)
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nModel: {summary['model']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"\nOverall Performance:")
    print(f"  Total Problems: {summary['overall']['total_problems']}")
    print(f"  Correct Solutions: {summary['overall']['correct_solutions']}")
    print(f"  Overall Accuracy: {summary['overall']['overall_accuracy']*100:.2f}%")
    print(f"  Average Reward: {summary['overall']['avg_reward']:.3f}")
    print(f"  Average Violations: {summary['overall']['avg_violations']:.2f}")
    print(f"  Average Generation Time: {summary['overall']['avg_generation_time']:.2f}s")
    
    print(f"\nPer-Disk Performance:")
    print(f"  {'Disks':<6} {'Correct/Total':<15} {'Accuracy':<12} {'Avg Reward':<12} {'Description':<50}")
    print(f"  {'-'*6} {'-'*15} {'-'*12} {'-'*12} {'-'*50}")
    
    for num_disks in sorted(summary['per_disk_count'].keys()):
        stats = summary['per_disk_count'][num_disks]
        correct_str = f"{stats['correct']}/{stats['num_trials']}"
        print(f"  {num_disks:<6} {correct_str:<15} "
              f"{stats['accuracy']*100:>6.1f}%      {stats['avg_reward']:>6.3f}      "
              f"{stats['description']:<50}")
    
    print("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("Waiting for vllm server to be ready...")
    
    # Test connection with retries
    max_retries = 12
    for i in range(max_retries):
        time.sleep(10)
        try:
            test_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                timeout=5.0
            )
            print(f"✓ Server is ready and responding!")
            break
        except Exception as e:
            print(f"  Retry {i+1}/{max_retries}: Waiting for server...")
            if i == max_retries - 1:
                print(f"ERROR: Server not responding after {max_retries} attempts")
                print(f"Last error: {e}")
                return
    
    # Run evaluation
    start_time = time.time()
    evaluation_data = run_evaluation()
    total_time = time.time() - start_time
    
    # Add total time to evaluation data
    evaluation_data['total_evaluation_time'] = total_time
    
    # Print final summary
    print_final_summary(evaluation_data)
    
    # Save results
    save_results(evaluation_data)
    
    print(f"\n✓ Evaluation complete! Total time: {total_time/60:.1f} minutes")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()