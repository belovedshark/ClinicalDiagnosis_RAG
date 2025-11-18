# INPUT vs OUTPUT - Visual Comparison

## Real Example from Your Clinical Cases

### INPUT: Raw Markdown File
**File:** `1---A-20-Year-Old-Woman-from-Sudan-With-Fever--_2022_Clinical-Cases-in-Tropi.md`

```markdown
# Page 11 A 20-Year-Old Woman from Sudan With Fever, Haemorrhage and Shock

## Clinical Presentation

### History
A 20-year-old housewife presents to a hospital in northern Uganda with a 2-day history 
of fever, severe asthenia, chest and abdominal pain, nausea, vomiting, diarrhoea...

### Clinical Findings
The patient is prostrate and semiconscious on admission. 

Vital signs:
- Temperature: 39.6°C (103.3°F)
- Blood pressure: 90/60 mmHg
- Pulse: 90 bpm
- Respiratory rate: 24 cycles per minute

Physical examination revealed abdominal tenderness, hepatosplenomegaly and bleeding 
from the gums.

## Questions

1. Is the patient's history and clinical presentation consistent with a haemorrhagic 
   fever (HF) syndrome?
2. What degree of nursing precautions need to be implemented?

## Discussion

This patient was seen during an outbreak of Ebola virus disease in northern Uganda, 
so the diagnosis was strongly suspected. She was admitted to the isolation ward...

[... continues with clinical discussion ...]

## The Case Continued

Intravenous fluids, broad-spectrum antibiotics and analgesics were begun on admission. 
Nevertheless, the patient's condition rapidly worsened...

## Summary Box

**Filoviral Diseases**

Ebola and Marburg virus disease are the two syndromes caused by filoviruses. 
Microvascular instability with capillary leak and impaired haemostasis, often including 
disseminated intravascular coagulation, are the pathogenic hallmarks.

- Four known pathogenic species of Ebola
- One species of Marburg virus
- Case fatality: 25% to 85%

## Further Reading

1. Blumberg L, Enria D, Bausch DG. Viral haemorrhagic fevers. In: Farrar J, editor...
2. Vetter P, Fischer 2nd WA, Schibler M, et al. Ebola Virus Shedding...

## Images

![figure](../images/.../image1.jpeg)
![figure](../images/.../image2.jpeg)
```

---

### EXTRACTION PROCESS

**Step 1:** Scan for table markers
- Pattern: `\bTABLE\s+\d+(?:\.\d+)?\b` (case-sensitive)
- Result: No "TABLE 1", "TABLE 2" found in this file
- Action: No tables to extract

**Step 2:** If tables were found:
- Find all start positions
- Search for end-signals: FIGURE, CHAPTER, REFERENCES, # 
- Extract text between start and end

---

### OUTPUT: Extracted Tables (if any found)

**File:** `1---A-20-Year-Old-Woman-from-Sudan-...-tables.md`

```markdown
[Empty or contains extracted tables separated by ====]

(In this case, the clinical case doesn't have formatted TABLE 1, TABLE 2 markers,
so nothing is extracted)
```

**File:** `1---A-20-...-tables.json`

```json
{
  "source": "Processed/markdown/1---A-20-Year-Old-Woman-from-Sudan.md",
  "tables": []
}
```

---

## Hypothetical Example WITH Tables

### INPUT: Markdown with table data

```markdown
## Clinical Findings

Patient assessment shows the following vital parameters:

TABLE 1: Vital Signs Measurements

| Time    | Temperature | BP (Systolic) | BP (Diastolic) | HR   | RR  |
|---------|-------------|---------------|----------------|------|-----|
| 06:00   | 39.6°C      | 90            | 60             | 90   | 24  |
| 12:00   | 39.2°C      | 92            | 62             | 92   | 26  |
| 18:00   | 38.8°C      | 95            | 65             | 88   | 23  |

Questions

1. What is the diagnosis?

FIGURE 1: Chest X-ray showing consolidation in right lower lobe
```

### EXTRACTION PROCESS (With tables)

```
STEP 1: Find table markers
  ✓ Found: "TABLE 1" at character position 245

STEP 2: Find table end
  ✓ Looking for end-signals after position 253...
  ✓ Found: "FIGURE" at character position 520
  → This is the earliest end-signal

STEP 3: Extract content
  Start position: 245
  End position: 520
  Extracted text: 275 characters, ~45 words

STEP 4: Create metadata
  id: 1
  start_line: 3
  end_line: 10
  word_count: 45
```

### OUTPUT: Extracted Tables

**File:** `example-tables.md`

```markdown
TABLE 1: Vital Signs Measurements

| Time    | Temperature | BP (Systolic) | BP (Diastolic) | HR   | RR  |
|---------|-------------|---------------|----------------|------|-----|
| 06:00   | 39.6°C      | 90            | 60             | 90   | 24  |
| 12:00   | 39.2°C      | 92            | 62             | 92   | 26  |
| 18:00   | 38.8°C      | 95            | 65             | 88   | 23  |
```

**File:** `example_table_1.md` (if --individual-files flag used)

```markdown
TABLE 1: Vital Signs Measurements

| Time    | Temperature | BP (Systolic) | BP (Diastolic) | HR   | RR  |
|---------|-------------|---------------|----------------|------|-----|
| 06:00   | 39.6°C      | 90            | 60             | 90   | 24  |
| 12:00   | 39.2°C      | 92            | 62             | 92   | 26  |
| 18:00   | 38.8°C      | 95            | 65             | 88   | 23  |
```

**File:** `example-tables.json` (if --output-json specified)

```json
{
  "source": "example.md",
  "tables": [
    {
      "id": 1,
      "start_pos": 245,
      "end_pos": 520,
      "start_line": 3,
      "end_line": 10,
      "word_count": 45,
      "excerpt": "TABLE 1: Vital Signs Measurements\n\n| Time | Temperature |..."
    }
  ]
}
```

---

## Multiple Tables Example

### INPUT: Markdown with multiple tables

```markdown
...

TABLE 1: Patient Demographics

| Age | Gender | Location | Duration |
|-----|--------|----------|----------|
| 20  | F      | Sudan    | 2 days   |

Questions

1. What tests are needed?

FIGURE 1: Lab results

TABLE 2: Laboratory Results

| Test | Value | Normal  |
|------|-------|---------|
| WBC  | 7.2   | 4-11    |
| RBC  | 4.8   | 4.5-5.5 |

CHAPTER 2: Next Case
```

### EXTRACTION PROCESS

```
Pass 1: Find all TABLE markers
  ✓ Found "TABLE 1" at pos 100
  ✓ Found "TABLE 2" at pos 400

Pass 2: Process TABLE 1
  - Start: pos 100
  - Search for end-signals after pos 110
  - Found: "FIGURE" at pos 300
  - Extract: chars 100-300 → Table 1 metadata

Pass 3: Process TABLE 2
  - Start: pos 400
  - Search for end-signals after pos 410
  - Found: "CHAPTER" at pos 600
  - Extract: chars 400-600 → Table 2 metadata
```

### OUTPUT: Multiple tables extracted

**File:** `example-tables.md`

```markdown
TABLE 1: Patient Demographics

| Age | Gender | Location | Duration |
|-----|--------|----------|----------|
| 20  | F      | Sudan    | 2 days   |

====
TABLE 2: Laboratory Results

| Test | Value | Normal  |
|------|-------|---------|
| WBC  | 7.2   | 4-11    |
| RBC  | 4.8   | 4.5-5.5 |
```

**File:** `example_table_1.md`

```markdown
TABLE 1: Patient Demographics

| Age | Gender | Location | Duration |
|-----|--------|----------|----------|
| 20  | F      | Sudan    | 2 days   |
```

**File:** `example_table_2.md`

```markdown
TABLE 2: Laboratory Results

| Test | Value | Normal  |
|------|-------|---------|
| WBC  | 7.2   | 4-11    |
| RBC  | 4.8   | 4.5-5.5 |
```

**File:** `example-tables.json`

```json
{
  "source": "example.md",
  "tables": [
    {
      "id": 1,
      "start_pos": 100,
      "end_pos": 300,
      "start_line": 5,
      "end_line": 12,
      "word_count": 18,
      "excerpt": "TABLE 1: Patient Demographics\n..."
    },
    {
      "id": 2,
      "start_pos": 400,
      "end_pos": 600,
      "start_line": 25,
      "end_line": 35,
      "word_count": 24,
      "excerpt": "TABLE 2: Laboratory Results\n..."
    }
  ]
}
```

---

## Summary: Input to Output Flow

```
INPUT FILES
    ↓
[extract_tables_md.py]
    ↓
┌─────────────────────────┐
│  Regex Pattern Match    │
│  - Find TABLE markers   │
│  - Find end-signals     │
│  - Extract boundaries   │
└─────────────────────────┘
    ↓
OUTPUTS:
├─ filename-tables.md (combined)
├─ filename_table_1.md (individual)
├─ filename_table_2.md (individual)
└─ summary.json (metadata)
```

---

## How to Use This Example

1. **With your actual files:**
   ```bash
   python scripts/extract_tables_md.py \
     --input Processed/markdown/1---A-20-Year-Old-Woman-from-Sudan.md \
     --output-dir ./tables_out \
     --debug
   ```

2. **To see what happens with tables:**
   - Your clinical cases may or may not have formatted TABLE markers
   - If they do: tables will be extracted
   - If they don't: output will be empty (0 tables found)

3. **Next step: Check output**
   ```bash
   ls -la tables_out/
   # See what files were created
   ```

4. **If no tables found:**
   - Check if your markdown has "TABLE" markers
   - Or use --debug to see what patterns are searched
   - Adjust --start-regex if using custom table headers
