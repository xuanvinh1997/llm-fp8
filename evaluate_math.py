import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_lm_eval(model_path: str, tasks: list, output_dir: str = None, batch_size: int = 4):
    """Run lm_eval on math tasks"""
    
    # Core math evaluation tasks (updated names)
    core_tasks = [
        "gsm8k",
        "math_qa",
        "mmlu_elementary_mathematics", 
        "mmlu_high_school_mathematics",
        "mmlu_abstract_algebra",
    ]
    
    # Use provided tasks or default to core tasks
    eval_tasks = tasks if tasks else core_tasks
    tasks_str = ",".join(eval_tasks)
    
    # Build lm_eval command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks_str,
        "--batch_size", str(batch_size),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    
    # Add output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd.extend(["--output_path", output_dir])
    
    # Add few-shot examples for math tasks
    cmd.extend(["--num_fewshot", "5"])
    
    print(f"Running evaluation command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Evaluation completed successfully!")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return None

def get_available_math_tasks():
    """Get list of available math-related tasks"""
    try:
        result = subprocess.run(["lm_eval", "--tasks", "list"], 
                              capture_output=True, text=True, check=True)
        
        # Filter for math-related tasks
        all_tasks = result.stdout.split('\n')
        math_tasks = []
        
        math_keywords = ['math', 'gsm', 'mmlu', 'algebra', 'geometry', 'arithmetic']
        
        for task in all_tasks:
            task = task.strip()
            if any(keyword in task.lower() for keyword in math_keywords):
                math_tasks.append(task)
        
        return math_tasks
    except subprocess.CalledProcessError:
        print("Could not retrieve task list")
        return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate math model with lm_eval")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--tasks", nargs="+", default=None, 
                       help="Specific tasks to evaluate (default: core math tasks)")
    parser.add_argument("--output_dir", default="./eval_results", 
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Evaluation batch size")
    parser.add_argument("--list_math_tasks", action="store_true",
                       help="List available math-related tasks")
    parser.add_argument("--minimal", action="store_true",
                       help="Run minimal set of core tasks only")
    
    args = parser.parse_args()
    
    # List available tasks if requested
    if args.list_math_tasks:
        print("Finding available math-related tasks...")
        math_tasks = get_available_math_tasks()
        print("Available math tasks:")
        for task in math_tasks:
            print(f"  - {task}")
        return
    
    # Define task sets
    if args.minimal:
        tasks = ["gsm8k"]  # Just GSM8K for quick testing
    elif args.tasks:
        tasks = args.tasks
    else:
        # Try common task names, fallback to minimal set
        candidate_tasks = [
            "gsm8k",
            "math_qa", 
            "mmlu_elementary_mathematics",
            "mmlu_high_school_mathematics",
            "mmlu_abstract_algebra",
        ]
        tasks = candidate_tasks
    
    print(f"Attempting to evaluate with tasks: {tasks}")
    
    # Run evaluation
    result = run_lm_eval(
        model_path=args.model_path,
        tasks=tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    if result:
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Evaluation failed. Try running with --list_math_tasks to see available tasks")
        print("Or use --minimal flag to test with just GSM8K")

if __name__ == "__main__":
    import torch
    main()