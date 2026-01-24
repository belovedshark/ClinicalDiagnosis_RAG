"""
Data preparation utilities for loading and processing WHO evaluation dataset.
"""
import json
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    return data


def prepare_evaluation_cases(jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare evaluation cases from WHO dataset.
    
    Args:
        jsonl_data: Raw data loaded from JSONL file
        
    Returns:
        List of prepared evaluation cases with standardized format
    """
    prepared_cases = []
    
    for idx, case in enumerate(jsonl_data):
        # Handle duplicate case_prompt field (appears twice in the data)
        case_prompt = case.get("case_prompt", "")
        if isinstance(case_prompt, str):
            # Single field case
            question = case_prompt
        elif isinstance(case_prompt, tuple) or isinstance(case_prompt, list):
            # Multiple fields (duplicate) - take the first one
            question = case_prompt[0] if case_prompt else ""
        else:
            question = str(case_prompt)
        
        prepared_case = {
            "case_id": f"who_case_{idx + 1:03d}",
            "question": question.strip(),
            "ground_truth": case.get("final_diagnosis", "").strip(),
            "diagnostic_reasoning": case.get("diagnostic_reasoning", "").strip()
        }
        
        prepared_cases.append(prepared_case)
    
    return prepared_cases


def validate_case(case: Dict[str, Any]) -> bool:
    """
    Validate that a case has all required fields.
    
    Args:
        case: Evaluation case dictionary
        
    Returns:
        True if case is valid, False otherwise
    """
    required_fields = ["case_id", "question", "ground_truth"]
    
    for field in required_fields:
        if field not in case or not case[field]:
            print(f"Warning: Case missing or empty field '{field}': {case.get('case_id', 'unknown')}")
            return False
    
    return True


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to: {output_path}")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint data if it exists.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint data or empty dict if file doesn't exist
    """
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Checkpoint file corrupted: {e}")
        return {}


def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_path: str):
    """
    Save checkpoint data.
    
    Args:
        checkpoint_data: Data to checkpoint
        checkpoint_path: Path to save checkpoint
    """
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
