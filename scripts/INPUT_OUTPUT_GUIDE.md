#!/usr/bin/env python3
"""
INPUT/OUTPUT GUIDE FOR EXTRACT_TABLES_MD.PY

This document shows exactly what the table extraction script expects as input
and what it produces as output.
"""

# ============================================================================
# SECTION 1: INPUT FORMAT
# ============================================================================

"""
INPUT: Markdown files from the PDF preprocessing pipeline

Location: Processed/markdown/
Files: 1---A-20-Year-Old-Woman-from-Sudan-With-Fever--_2022_Clinical-Cases-in-Tropi.md
       (and 94+ similar clinical case files)

STRUCTURE OF INPUT MARKDOWN:
- Page markers (# Page 1, # Page 2, etc.)
- Section headers (## Clinical Presentation, ## Questions, etc.)
- Narrative text (patient history, clinical findings, discussion)
- Occasionally: TABLE markers (TABLE 1, TABLE 2, etc.)
- References and further reading
- Image references (![figure](path/to/image.jpg))

KEY CHARACTERISTICS:
✓ Plain text content
✓ May contain multiple tables with "TABLE 1", "TABLE 2" markers
✓ Tables embedded within narrative flow
✓ May have end-signal patterns: FIGURE, CHAPTER, REFERENCES
✓ UTF-8 encoded

EXAMPLE INPUT SNIPPET:
"""

input_example = """
## Clinical Findings

The patient is prostrate and semiconscious on admission. Vital signs:
temperature 39.6°C, blood pressure 90/60mmHg, pulse 90bpm.

Physical examination revealed abdominal tenderness, hepatosplenomegaly
and bleeding from the gums.

TABLE 1: Vital Signs Measurements

Patient ID | Temperature | BP (Systolic) | BP (Diastolic) | Pulse
-----------|-------------|--------------|----------------|-------
Patient 1  | 39.6°C      | 90           | 60             | 90
Patient 2  | 38.2°C      | 110          | 70             | 95
Patient 3  | 40.1°C      | 85           | 55             | 88

Questions

1. Is the patient's history consistent with hemorrhagic fever?
2. What precautions need to be implemented?

FIGURE 1: Chest X-ray showing clear lungs
...
"""

# ============================================================================
# SECTION 2: EXTRACTION PROCESS
# ============================================================================

"""
HOW THE EXTRACTION WORKS:

STEP 1: FIND TABLE STARTS
Pattern: r"\\bTABLE\\s+\\d+(?:\\.\\d+)?\\b"
Matches: "TABLE 1", "TABLE 2", "TABLE 2.1", "TABLE 2.1.1", etc.
Case: Sensitive (only matches capitalized "TABLE")

Example matches in text:
  "TABLE 1: Vital Signs Measurements" → matches "TABLE 1"
  "TABLE 2.1: Test Results" → matches "TABLE 2.1"


STEP 2: FIND TABLE ENDS
Patterns (case-insensitive):
  - "FIGURE" (start of next section)
  - "CHAPTER" (new chapter marker)
  - "REFERENCES" (end of content)
  - "#" (page or section marker)
  - Or another "TABLE" (next table)

Strategy: Find the EARLIEST match (closest to table start)
This prevents accidentally including multiple tables

Example:
  Text: "...TABLE 1...FIGURE 1... ...CHAPTER 2..."
  Table 1 extracted from: TABLE 1 to FIGURE 1 (FIGURE is closest)


STEP 3: FALLBACK MECHANISM
If no end-signal found:
  Extract first N words (default: 500 words)
  This handles tables that end at document end

Example:
  max_words = 500
  If 500 words extracted before any end-signal, stop there


STEP 4: CREATE METADATA
For each extracted table, store:
  - id: sequence number (1, 2, 3, ...)
  - start_pos: character offset in original text
  - end_pos: character offset in original text
  - start_line: line number (1-indexed)
  - end_line: line number (1-indexed)
  - word_count: approximate word count
  - excerpt: actual table content
"""

# ============================================================================
# SECTION 3: OUTPUT FORMATS
# ============================================================================

"""
OUTPUT 1: COMBINED MARKDOWN FILE
Filename: <input_name>-tables.md
Location: output-dir/ (or same as input if not specified)

Format: All extracted tables in one file, separated by "===="

EXAMPLE OUTPUT:
"""

output_combined_example = """
TABLE 1: Vital Signs Measurements

Patient ID | Temperature | BP (Systolic) | BP (Diastolic) | Pulse
-----------|-------------|--------------|----------------|-------
Patient 1  | 39.6°C      | 90           | 60             | 90
Patient 2  | 38.2°C      | 110          | 70             | 95
Patient 3  | 40.1°C      | 85           | 55             | 88

====
TABLE 2: Laboratory Results

Test | Patient 1 | Patient 2 | Patient 3 | Normal Range
-----|-----------|-----------|-----------|---------------
WBC  | 7.2       | 12.1      | 4.5       | 4.5-11.0
RBC  | 4.8       | 3.9       | 5.2       | 4.5-5.5
Hgb  | 14.2      | 11.5      | 15.1      | 13.5-17.5

====
TABLE 3: Clinical Assessment Scores

Patient | SOFA | qSOFA | Mortality Risk
--------|------|-------|----------------
Pt1     | 2    | 1     | Low
Pt2     | 6    | 2     | High
Pt3     | 1    | 0     | Very Low
"""

"""
OUTPUT 2: INDIVIDUAL MARKDOWN FILES (optional --individual-files)
Filenames: <input_name>_table_1.md, <input_name>_table_2.md, etc.
Location: output-dir/

Each file contains a single table excerpt
Useful for: per-table processing, parallel embedding, version control

Example files:
  input_table_1.md ← Contains only TABLE 1
  input_table_2.md ← Contains only TABLE 2
"""

"""
OUTPUT 3: JSON SUMMARY (optional --output-json)
Filename: <output_name>.json or <input_name>-tables.json
Format: Structured metadata about extracted tables

EXAMPLE OUTPUT:
"""

output_json_example = """{
  "source": "Processed/markdown/case1.md",
  "tables": [
    {
      "id": 1,
      "start_pos": 1250,
      "end_pos": 2500,
      "start_line": 45,
      "end_line": 87,
      "word_count": 234,
      "excerpt": "TABLE 1: Vital Signs Measurements\\n..."
    },
    {
      "id": 2,
      "start_pos": 3100,
      "end_pos": 4200,
      "start_line": 100,
      "end_line": 130,
      "word_count": 156,
      "excerpt": "TABLE 2: Laboratory Results\\n..."
    }
  ]
}"""

# ============================================================================
# SECTION 4: USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: EXTRACT FROM SINGLE FILE
Command:
  python scripts/extract_tables_md.py \\
    --input Processed/markdown/1---A-20-Year-Old-Woman-from-Sudan.md \\
    --output-dir ./tables_output

Output:
  ./tables_output/1---A-20-Year-Old-Woman-from-Sudan-tables.md


EXAMPLE 2: EXTRACT FROM ALL FILES IN DIRECTORY
Command:
  python scripts/extract_tables_md.py \\
    --input Processed/markdown/ \\
    --output-dir ./all_tables

Output:
  ./all_tables/1---A-20-Year-Old-Woman-from-Sudan-tables.md
  ./all_tables/2---A-7-Year-Old-Girl-from-Peru-tables.md
  ./all_tables/3---A-26-Year-Old-Woman-from-Malawi-tables.md
  ... (one for each input file)


EXAMPLE 3: EXTRACT WITH INDIVIDUAL FILES + JSON SUMMARY
Command:
  python scripts/extract_tables_md.py \\
    --input Processed/markdown/ \\
    --output-dir ./extraction_results \\
    --individual-files \\
    --output-json ./extraction_results/all_tables_summary.json \\
    --debug

Output:
  ./extraction_results/
    ├── 1---A-20-...md                    # combined tables
    ├── 1---A-20-..._table_1.md           # individual table
    ├── 1---A-20-..._table_2.md
    ├── 2---A-7-...md                     # combined tables
    ├── 2---A-7-..._table_1.md            # individual table
    ├── all_tables_summary.json           # combined JSON
    └── (debug output in terminal)


EXAMPLE 4: CUSTOM REGEX PATTERNS
Command:
  python scripts/extract_tables_md.py \\
    --input data.md \\
    --start-regex "(?i)data\\s+table\\s+\\d+" \\
    --end-signals "FINDINGS" "RESULTS" "CONCLUSION" \\
    --max-words 2000

Use case: When your markdown has non-standard table markers
  - Matches: "Data Table 1", "DATA TABLE 5" (case-insensitive)
  - Custom end signals
  - Extract up to 2000 words per table


EXAMPLE 5: DEBUG MODE
Command:
  python scripts/extract_tables_md.py \\
    --input file.md \\
    --debug

Terminal output will show:
  [debug] Found 3 start match(es)
  [debug] start#1: pos=100..108 -> 'TABLE 1'
  [debug] start#2: pos=500..508 -> 'TABLE 2'
  [debug] start#3: pos=1200..1208 -> 'TABLE 3'
  [debug] end-signal pattern '(?i)\\bFIGURE\\b' matched at 200..206 -> 'FIGURE'
  [debug] selected end at 200 using pattern (?i)\\bFIGURE\\b
  [debug] excerpt for start#1: chars 100..200, approx words 45
  ...
"""

# ============================================================================
# SECTION 5: EXPECTED FILE SIZES & PERFORMANCE
# ============================================================================

"""
FILE SIZE STATISTICS:

INPUT (Processed markdown files):
  - Average: 10-50 KB per file
  - Total corpus: ~4-5 MB
  - Format: Plain text UTF-8

OUTPUT (Extracted tables):
  - Combined markdown: 2-10 KB per file (typically 10-50% of input)
  - Individual markdown: 0.5-5 KB per table
  - JSON summary: 1-5 KB per file

PERFORMANCE:
  - Single file extraction: ~50-200ms
  - 100 files batch: ~5-20 seconds
  - Bottleneck: Regex matching on very large files
  - Memory: Minimal (< 100MB for full corpus)

TYPICAL SCENARIO:
  Input:  94 clinical case files (~5 MB total)
  Output: ~200-300 extracted tables (~1-2 MB)
  Time:   ~10-15 seconds total
"""

# ============================================================================
# SECTION 6: COMMON ISSUES & TROUBLESHOOTING
# ============================================================================

"""
ISSUE 1: No tables extracted from a file
Cause 1: File has no "TABLE" markers (only other content)
  → Check if your files actually contain table headers
  → Use --debug to see what patterns are being searched

Cause 2: TABLE markers are lowercase (e.g., "table 1")
  → Default regex is case-sensitive for start pattern
  → Use custom: --start-regex "(?i)\\bTABLE\\s+\\d+"

Cause 3: Table markers are in unusual format
  → Use --debug and inspect output
  → Customize --start-regex to match your format


ISSUE 2: Extracted tables are incomplete
Cause: End-signal not detected, fallback to max_words
  → Increase --max-words to extract more content
  → Or add end-signal patterns: --end-signals "PATTERN1" "PATTERN2"


ISSUE 3: Multiple tables merged into one
Cause: Next "TABLE" marker not recognized as end-signal
  → Already in default end_signals: r"(?i)\\bTABLE\\b"
  → Check if table header is in unusual format

Solution: Use --debug to see what was matched


ISSUE 4: JSON output is empty
Cause: Per-file JSON not created
  → Ensure --output-json path is writable
  → Check directory permissions
  → Use full absolute path if relative path fails
"""

# ============================================================================
# SECTION 7: INTEGRATION WITH EMBEDDING PIPELINE
# ============================================================================

"""
WORKFLOW INTEGRATION:

Step 1: PDF → Markdown (preprocessing_pdf.py)
  Input:  PDFs in SourceMedicalRecords/
  Output: Markdown in Processed/markdown/

Step 2: Markdown → Extracted Tables (extract_tables_md.py)  ← YOU ARE HERE
  Input:  Markdown from step 1
  Output: Extracted tables in output-dir/

Step 3: Tables → Embeddings (embedding_clip.py)
  Input:  Extracted tables from step 2
  Output: CLIP embeddings in Qdrant

Step 4: Query & RAG
  Query tables + context for LLM responses


BENEFITS OF THIS APPROACH:
✓ Separates structured data (tables) from narrative
✓ Enables specialized processing for tabular content
✓ Improves retrieval for numerical/data queries
✓ Supports A/B testing of different indexing strategies


NEXT STEPS AFTER EXTRACTION:
1. Review extracted tables for quality
2. Adjust regex patterns if needed
3. Feed extracted tables to embedding pipeline
4. Index in Qdrant with table-specific metadata
5. Monitor retrieval performance
"""

# ============================================================================
# SECTION 8: REAL EXAMPLE - STEP BY STEP
# ============================================================================

"""
CONCRETE EXAMPLE WALKTHROUGH:

INPUT FILE: "1---A-20-Year-Old-Woman-from-Sudan.md"
Content snippet:
  "...patient is prostrate and semiconscious...
   
   TABLE 1: Vital Signs
   
   Patient ID | Temp | BP Systolic
   1          | 39.6 | 90
   2          | 38.2 | 110
   
   Questions
   
   1. Is the diagnosis correct?
   
   FIGURE 1: Chest X-ray"


EXTRACTION PROCESS:
1. Read entire file
2. Search for pattern: r"\\bTABLE\\s+\\d+(?:\\.\\d+)?\\b"
   → Found: "TABLE 1" at position 245
   
3. Search for end-signals starting at position 253:
   - Check "(?i)\\bTABLE\\b" → not found
   - Check "(?i)\\bFIGURE\\b" → FOUND at position 420
   
4. Extract text from position 245 to 420
5. Store result:
   {
     "id": 1,
     "start_pos": 245,
     "end_pos": 420,
     "start_line": 15,
     "end_line": 28,
     "word_count": 45,
     "excerpt": "TABLE 1: Vital Signs...
                Patient ID | Temp | BP Systolic
                1          | 39.6 | 90
                2          | 38.2 | 110"
   }

6. Write output:
   - "1---A-20-...-tables.md" → includes table excerpt
   - "1---A-20-..._table_1.md" (if --individual-files)
   - "summary.json" (if --output-json)


VERIFICATION:
✓ Table extracted from correct position
✓ Boundaries correctly identified
✓ Metadata accurately recorded
✓ Output files created successfully
"""

# ============================================================================
# SECTION 9: COMMAND REFERENCE CHEAT SHEET
# ============================================================================

commands_reference = """
QUICK REFERENCE - COPY & PASTE COMMANDS:

1. Extract from single file (basic):
   python scripts/extract_tables_md.py --input file.md --output-dir ./out

2. Extract from directory (all files):
   python scripts/extract_tables_md.py --input ./input_dir --output-dir ./out

3. Extract with individual files:
   python scripts/extract_tables_md.py --input ./md --output-dir ./out --individual-files

4. Extract with debug output:
   python scripts/extract_tables_md.py --input file.md --debug

5. Extract with JSON summary:
   python scripts/extract_tables_md.py --input ./md --output-json ./summary.json --output-dir ./out

6. Custom patterns (for non-standard formats):
   python scripts/extract_tables_md.py --input file.md --start-regex "(?i)data\\s+table" --end-signals "SECTION" "CHAPTER"

7. Increase word limit for long tables:
   python scripts/extract_tables_md.py --input file.md --max-words 2000

8. Full example (all options):
   python scripts/extract_tables_md.py \\
     --input ./Processed/markdown \\
     --output-dir ./extracted_tables \\
     --output-json ./summary.json \\
     --individual-files \\
     --max-words 1000 \\
     --debug

---

COMMAND OPTION REFERENCE:
  --input PATH              Path to markdown file or directory (REQUIRED)
  --output-dir PATH         Directory for output files (default: same as input)
  --output-json PATH        Write JSON summary to this file (optional)
  --individual-files        Write one .md per table (optional)
  --max-words N             Max words per table (default: 500)
  --start-regex PATTERN     Custom start pattern (case-sensitive)
  --end-signals PAT1 PAT2   Custom end patterns (space-separated)
  --case-sensitive          Make all patterns case-sensitive
  --include-end             Include the end-signal line in excerpt
  --debug                   Print debug information
"""

print(commands_reference)

if __name__ == "__main__":
    print("\n✓ This guide explains input/output formats for extract_tables_md.py")
    print("✓ See the embedded docstrings for detailed examples")
    print("✓ Use as reference when running the extraction script")
