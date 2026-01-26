"""
Sample clinical questions for demo purposes.

These cases are curated to demonstrate the differences between model variants:
- Simple cases where all models likely agree
- Complex cases where RAG/finetuning should help
- Edge cases to highlight model differences
"""

SAMPLE_QUESTIONS = [
    {
        "id": 1,
        "name": "Classic Dengue Fever",
        "question": "A 24-year-old man presents with high fever, retro-orbital pain, severe muscle and joint pain, and rash after visiting a tropical city with mosquito exposure.",
        "ground_truth": "Dengue fever",
        "difficulty": "easy",
        "notes": "Classic presentation - most models should get this correct"
    },
    {
        "id": 2,
        "name": "Malaria with Cyclical Fever",
        "question": "A 33-year-old traveler presents with cyclical fever, chills, sweating, and headache after visiting a malaria-endemic region.",
        "ground_truth": "Malaria",
        "difficulty": "easy",
        "notes": "Typical malaria presentation with travel history"
    },
    {
        "id": 3,
        "name": "Severe Dengue Warning Signs",
        "question": "A 42-year-old woman presents with persistent fever, severe abdominal pain, vomiting, and bleeding gums after recent dengue infection.",
        "ground_truth": "Severe dengue",
        "difficulty": "medium",
        "notes": "Requires recognizing progression from dengue to severe dengue"
    },
    {
        "id": 4,
        "name": "Chikungunya vs Dengue",
        "question": "A 29-year-old woman presents with sudden fever and severe joint pain affecting hands and ankles after mosquito exposure.",
        "ground_truth": "Chikungunya",
        "difficulty": "medium",
        "notes": "Distinguished by prominent arthralgia - RAG context should help"
    },
    {
        "id": 5,
        "name": "Visceral Leishmaniasis",
        "question": "A 7-year-old boy from a rural area presents with prolonged fever, weight loss, and massive splenomegaly.",
        "ground_truth": "Leishmaniasis (visceral)",
        "difficulty": "hard",
        "notes": "Less common disease - fine-tuned model or RAG may perform better"
    },
    {
        "id": 6,
        "name": "Zika in Pregnancy",
        "question": "A pregnant woman develops low-grade fever and rash after travel to a Zika-endemic area.",
        "ground_truth": "Zika virus infection",
        "difficulty": "medium",
        "notes": "Important to recognize pregnancy context"
    },
    {
        "id": 7,
        "name": "Neurocysticercosis",
        "question": "A 30-year-old man from a rural community presents with new-onset seizures and a history of eating undercooked pork.",
        "ground_truth": "Neurocysticercosis",
        "difficulty": "hard",
        "notes": "Requires connecting dietary history to parasitic infection"
    },
    {
        "id": 8,
        "name": "Leptospirosis",
        "question": "A 28-year-old farmer presents with high fever, severe headache, muscle pain, and jaundice after wading through floodwaters.",
        "ground_truth": "Leptospirosis",
        "difficulty": "hard",
        "notes": "Occupational and environmental exposure are key clues"
    },
    {
        "id": 9,
        "name": "Scrub Typhus",
        "question": "A 35-year-old hiker presents with fever, headache, and a black eschar lesion on the leg after trekking through tall grass.",
        "ground_truth": "Scrub typhus",
        "difficulty": "hard",
        "notes": "Eschar is pathognomonic - RAG should help identify"
    },
    {
        "id": 10,
        "name": "African Trypanosomiasis",
        "question": "A 40-year-old safari guide presents with intermittent fever, headache, and excessive daytime sleepiness after a tsetse fly bite.",
        "ground_truth": "African trypanosomiasis",
        "difficulty": "hard",
        "notes": "Rare disease with specific vector - tests model knowledge limits"
    },
]


def get_sample_question(question_id: int = None, difficulty: str = None) -> dict:
    """
    Get a sample question by ID or difficulty level.
    
    Args:
        question_id: Specific question ID (1-10)
        difficulty: Filter by difficulty ("easy", "medium", "hard")
        
    Returns:
        Sample question dictionary
    """
    if question_id is not None:
        for q in SAMPLE_QUESTIONS:
            if q["id"] == question_id:
                return q
        raise ValueError(f"No question with ID {question_id}. Valid IDs: 1-{len(SAMPLE_QUESTIONS)}")
    
    if difficulty is not None:
        filtered = [q for q in SAMPLE_QUESTIONS if q["difficulty"] == difficulty]
        if filtered:
            return filtered[0]
        raise ValueError(f"No questions with difficulty '{difficulty}'")
    
    # Default: return first question
    return SAMPLE_QUESTIONS[0]


def list_sample_questions():
    """Print a formatted list of all sample questions."""
    print("\nAvailable Sample Questions:")
    print("=" * 60)
    for q in SAMPLE_QUESTIONS:
        print(f"  [{q['id']:2d}] {q['name']} ({q['difficulty']})")
        print(f"      Ground truth: {q['ground_truth']}")
    print("=" * 60)
