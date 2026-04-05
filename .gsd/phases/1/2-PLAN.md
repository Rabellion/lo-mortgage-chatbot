---
phase: 1
plan: 2
wave: 2
---

# Plan 1.2: CFPB JSON Ingestion Loader

## Objective
LangChain has no built-in loader for the CFPB JSON format. Write a custom loader that extracts complaint narratives from the CFPB JSON and converts them into LangChain Document objects so the standard chunking pipeline can process them identically to PDFs.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- data/raw/cfpb_mortgage_complaints.json (must exist — run Plan 1.1 first)

## Tasks

<task type="auto">
  <name>Write CFPBComplaintLoader</name>
  <files>
    src/data/loaders/cfpb.py
    src/data/loaders/__init__.py
  </files>
  <action>
    Write src/data/loaders/cfpb.py containing CFPBComplaintLoader class that:
    - Takes a file path to the CFPB JSON file
    - Parses the nested structure: data["hits"]["hits"] → each hit["_source"]
    - Extracts: consumer_complaint_narrative (skip if None or empty)
    - Builds LangChain Document with:
        page_content = complaint narrative text
        metadata = {
          "source_type": "cfpb_complaint",
          "product": hit["_source"].get("product", ""),
          "sub_product": hit["_source"].get("sub_product", ""),
          "issue": hit["_source"].get("issue", ""),
          "date_received": hit["_source"].get("date_received", ""),
          "complaint_id": hit["_source"].get("complaint_id", ""),
        }
    - Returns list of Documents (skip entries with no narrative)
    - Print count of loaded vs skipped entries using Rich

    DO NOT fabricate data — only use what exists in the JSON structure.
    DO NOT use any field that might contain PII beyond what CFPB already makes public.
  </action>
  <verify>python -c "from src.data.loaders.cfpb import CFPBComplaintLoader; docs = CFPBComplaintLoader('data/raw/cfpb_mortgage_complaints.json').load(); print(len(docs), 'docs loaded'); assert len(docs) > 0"</verify>
  <done>Prints positive doc count without error; each doc has non-empty page_content and source_type metadata</done>
</task>
