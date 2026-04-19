import os
import glob
import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match common patterns:
    # "" or "" or "Phase X " or "Phase X: " or ""
    # Examples: 
    # "Hybrid Retrieval" -> "Hybrid Retrieval"
    # "Embeddings" -> "Embeddings"
    # "Orchestrator:" -> "Orchestrator:"
    # "(schema)" -> "(Schema)" or just let the regex take the "" out to "schema"
    # "(shared with settings)" -> "(shared with settings)"
    
    # 1. Broad replacements for headings/prefixes
    content = re.sub(r'(?i)Phase \d+(?:\+\d+)?\s*[-—:]\s*', '', content)
    
    # 2. In-sentence replacements
    content = re.sub(r'(?i)Phase \d+(?:\+\d+)?\s*', '', content)
    
    # 3. Specific Swagger Tags cleanup (e.g., tags=["Embeddings"] -> tags=["Embeddings"])
    # The first regex already stripped ``, so `tags=["Embeddings"]` should be left.
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    skip_dirs = ['venv', '.venv', '__pycache__', 'site-packages']
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for f in files:
            if f.endswith('.py') or f.endswith('.md') or f.endswith('.txt'):
                process_file(os.path.join(root, f))

if __name__ == '__main__':
    main()
