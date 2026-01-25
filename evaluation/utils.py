"""
Shared utilities for the evaluation framework.

Provides common functions for data loading, saving, and checkpointing
used by all evaluators.
"""
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


# =============================================================================
# DIAGNOSIS MATCHING - Hybrid Approach (Alias Mapping + Semantic Similarity)
# =============================================================================

# Canonical diagnosis -> list of equivalent terms/aliases
# Add more mappings as needed based on your domain
DIAGNOSIS_ALIASES = {
    # Malaria and its species
    "malaria": [
        "p. falciparum", "p. vivax", "p. ovale", "p. malariae", "p. knowlesi",
        "plasmodium falciparum", "plasmodium vivax", "plasmodium ovale", 
        "plasmodium malariae", "plasmodium knowlesi", "plasmodium",
        "falciparum malaria", "vivax malaria", "cerebral malaria",
        "malaria infection", "malarial infection"
    ],
    # Dengue variants
    "dengue fever": [
        "dengue", "denv", "dengue virus", "dengue virus infection",
        "classic dengue", "dengue infection"
    ],
    "severe dengue": [
        "dengue hemorrhagic fever", "dhf", "dengue shock syndrome", "dss",
        "hemorrhagic dengue"
    ],
    # Chikungunya
    "chikungunya": [
        "chikungunya fever", "chikungunya virus", "chikungunya infection",
        "chikv", "chik"
    ],
    # Zika
    "zika virus infection": [
        "zika", "zika virus", "zika fever", "zikv"
    ],
    # Yellow fever
    "yellow fever": [
        "yellow fever virus", "yfv", "yellow fever infection"
    ],
    # Leishmaniasis
    "leishmaniasis (cutaneous)": [
        "cutaneous leishmaniasis", "skin leishmaniasis", "oriental sore",
        "cl", "localized cutaneous leishmaniasis"
    ],
    "leishmaniasis (visceral)": [
        "visceral leishmaniasis", "kala-azar", "kala azar", "vl",
        "black fever", "dumdum fever"
    ],
    # Schistosomiasis
    "schistosomiasis": [
        "bilharzia", "bilharziasis", "snail fever", "schistosoma",
        "schistosoma mansoni", "schistosoma haematobium", "schistosoma japonicum"
    ],
    # Filariasis
    "lymphatic filariasis": [
        "elephantiasis", "filariasis", "wuchereria bancrofti",
        "brugia malayi", "lf"
    ],
    # Trypanosomiasis
    "african trypanosomiasis": [
        "sleeping sickness", "african sleeping sickness", "trypanosomiasis",
        "trypanosoma brucei", "hat", "human african trypanosomiasis"
    ],
    # Helminthiasis
    "soil-transmitted helminthiasis": [
        "sth", "intestinal worms", "geohelminths", "soil transmitted helminths"
    ],
    "hookworm infection": [
        "hookworm", "ancylostomiasis", "necatoriasis", "ancylostoma",
        "necator americanus", "ancylostoma duodenale"
    ],
    "strongyloidiasis": [
        "strongyloides", "strongyloides stercoralis", "threadworm infection"
    ],
    "trichuriasis": [
        "whipworm", "whipworm infection", "trichuris trichiura"
    ],
    # Amoebiasis
    "amoebiasis": [
        "amebiasis", "amoebic dysentery", "entamoeba histolytica",
        "intestinal amoebiasis", "amoebic colitis"
    ],
    # Cholera
    "cholera": [
        "vibrio cholerae", "cholera infection", "asiatic cholera"
    ],
    # Leptospirosis
    "leptospirosis": [
        "weil's disease", "weils disease", "leptospira", "rat fever",
        "canicola fever", "leptospiral infection"
    ],
    # Scrub typhus
    "scrub typhus": [
        "tsutsugamushi disease", "orientia tsutsugamushi", "mite typhus",
        "bush typhus"
    ],
    # Relapsing fever
    "relapsing fever (tropical form)": [
        "relapsing fever", "borrelia recurrentis", "tick-borne relapsing fever",
        "louse-borne relapsing fever", "epidemic relapsing fever"
    ],
    # Neurocysticercosis
    "neurocysticercosis": [
        "ncc", "cysticercosis", "taenia solium", "pork tapeworm",
        "cerebral cysticercosis", "brain cysticercosis"
    ],
}

# Build reverse mapping for fast lookup
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in DIAGNOSIS_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical.lower()
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical.lower()


def normalize_diagnosis(answer: str) -> str:
    """
    Normalize a diagnosis to its canonical form using alias mapping.
    
    Args:
        answer: Raw diagnosis string
        
    Returns:
        Canonical diagnosis name (lowercase) or original if no mapping found
    """
    if not answer:
        return ""
    
    answer_lower = answer.lower().strip()
    
    # Direct lookup in alias mapping
    if answer_lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[answer_lower]
    
    # Check if any alias is contained in the answer (for partial matches)
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias in answer_lower or answer_lower in alias:
            return canonical
    
    return answer_lower


class SemanticMatcher:
    """
    Semantic similarity matcher using sentence embeddings.
    Used as fallback when alias mapping doesn't find a match.
    """
    
    def __init__(self, model_name: str = "thenlper/gte-small", threshold: float = 0.85):
        """
        Initialize the semantic matcher.
        
        Args:
            model_name: HuggingFace model name for embeddings
            threshold: Similarity threshold for considering a match (0.0-1.0)
        """
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._model = self._model.to('mps')
                    
            except ImportError:
                print("Warning: transformers not available for semantic matching")
                return False
        return True
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text string."""
        if not self._load_model():
            return None
        
        import torch
        
        inputs = self._tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        # Move to same device as model
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def is_match(self, answer: str, ground_truth: str) -> Tuple[bool, float]:
        """
        Check if answer semantically matches ground truth.
        
        Args:
            answer: Model's answer
            ground_truth: Expected answer
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = self.compute_similarity(answer, ground_truth)
        return similarity >= self.threshold, similarity


# Global semantic matcher instance (lazy-loaded)
_semantic_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher(threshold: float = 0.85) -> SemanticMatcher:
    """Get or create the global semantic matcher instance."""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticMatcher(threshold=threshold)
    return _semantic_matcher


def is_diagnosis_match(
    answer: str, 
    ground_truth: str, 
    use_semantic: bool = True,
    semantic_threshold: float = 0.85
) -> Tuple[bool, str, float]:
    """
    Check if model answer matches ground truth using hybrid approach.
    
    Matching strategy:
    1. Exact match (after normalization)
    2. Alias mapping match
    3. Semantic similarity match (if enabled)
    
    Args:
        answer: Model's diagnosis answer
        ground_truth: Expected diagnosis
        use_semantic: Whether to use semantic similarity as fallback
        semantic_threshold: Threshold for semantic similarity (0.0-1.0)
        
    Returns:
        Tuple of (is_match, match_type, confidence_score)
        - match_type: "exact", "alias", "semantic", or "none"
        - confidence_score: 1.0 for exact/alias, similarity score for semantic
    """
    if not answer or not ground_truth:
        return False, "none", 0.0
    
    answer_clean = answer.lower().strip()
    ground_truth_clean = ground_truth.lower().strip()
    
    # 1. Exact match
    if answer_clean == ground_truth_clean:
        return True, "exact", 1.0
    
    # 2. Alias mapping match
    answer_normalized = normalize_diagnosis(answer)
    ground_truth_normalized = normalize_diagnosis(ground_truth)
    
    if answer_normalized == ground_truth_normalized:
        return True, "alias", 1.0
    
    # 3. Semantic similarity match (fallback)
    if use_semantic:
        try:
            matcher = get_semantic_matcher(threshold=semantic_threshold)
            is_match, similarity = matcher.is_match(answer, ground_truth)
            if is_match:
                return True, "semantic", similarity
            # Return the similarity score even if not a match
            return False, "none", similarity
        except Exception as e:
            print(f"Warning: Semantic matching failed: {e}")
    
    return False, "none", 0.0


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
    Prepare evaluation cases from raw JSONL data.
    
    Args:
        jsonl_data: Raw data loaded from JSONL file
        
    Returns:
        List of prepared evaluation cases with standardized format
    """
    prepared_cases = []
    
    for idx, case in enumerate(jsonl_data):
        # Handle case_prompt field
        case_prompt = case.get("case_prompt", "")
        if isinstance(case_prompt, str):
            question = case_prompt
        elif isinstance(case_prompt, (tuple, list)):
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


def clean_diagnosis_output(answer: str) -> str:
    """
    Clean up the generated diagnosis output.
    
    Args:
        answer: Raw model output
        
    Returns:
        Cleaned diagnosis string
    """
    answer = answer.strip()
    
    # If answer is too long, extract first meaningful part
    if len(answer) > 100:
        answer = answer.split('\n')[0].strip()
        if '.' in answer:
            answer = answer.split('.')[0].strip()
    
    # Remove markdown formatting
    answer = answer.replace('**', '').replace('*', '')
    
    # Split into lines and filter out separators
    lines = []
    for line in answer.split('\n'):
        line = line.strip()
        if not line:
            continue
        if set(line) <= {'-', '_', '=', '*', '#'}:
            continue
        if len(line) < 3:
            continue
        lines.append(line)
    
    if lines:
        answer = lines[0]
    else:
        answer = "Unable to determine diagnosis"
    
    # Remove trailing punctuation
    if ' - ' in answer:
        answer = answer.split(' - ')[0].strip()
    if answer.endswith('.'):
        answer = answer[:-1]
    if answer.endswith(':'):
        answer = answer[:-1]
    
    return answer
