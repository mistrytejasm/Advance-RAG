"""
View Extracted Document — Renders processed JSON chunks as a structured,
readable document that mirrors the original PDF layout.

Usage:
    python view_extracted.py <document_id>
    python view_extracted.py               # Lists all available documents

Example:
    python view_extracted.py 2a9396a0-9d70-4e09-8a17-ac73814f3904
"""

import json
import sys
import os
import re
import io

# Fix Windows terminal encoding — force UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROCESSED_DIR = "data/processed"


def clean_pua_chars(text):
    """Replace common Private Use Area (PUA) Unicode characters with readable equivalents."""
    replacements = {
        "\ue081": "(",
        "\ue082": ")",
        "\ue083": " ✅",
        "\ue084": "",
        "\ue085": "'",
        "\ue086": '"',
        "\ue087": '"',
        "\ue088": "-",
        "\ue089": "•",
        "\ue08a": "→",
        "\ue08b": "—",
        "\ue08c": "…",
        "\ue08d": "'",
        "\ue08e": "'",
        "\ue08f": "–",
        "\ue090": "×",
        "\ue091": "÷",
        "\ue092": ":",
    }
    for pua, replacement in replacements.items():
        text = text.replace(pua, replacement)
    # Remove any remaining PUA characters (U+E000 to U+F8FF)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)
    return text


def list_documents():
    """List all processed documents."""
    if not os.path.isdir(PROCESSED_DIR):
        print("No processed documents found.")
        return

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".json")]
    if not files:
        print("No processed documents found.")
        return

    print("=" * 60)
    print("  AVAILABLE DOCUMENTS")
    print("=" * 60)
    for f in sorted(files):
        filepath = os.path.join(PROCESSED_DIR, f)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        doc_id = data.get("document_id", f.replace(".json", ""))
        chunk_count = len(data.get("chunks", []))
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  📄 {doc_id}")
        print(f"     Chunks: {chunk_count}  |  Size: {size_kb:.1f} KB")
        print()


def render_document(doc_id):
    """Render a processed document in structured readable format."""
    filepath = os.path.join(PROCESSED_DIR, f"{doc_id}.json")
    if not os.path.exists(filepath):
        print(f"❌ Document not found: {filepath}")
        print("Run without arguments to list available documents.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        print("⚠️  Document has no extracted chunks.")
        return

    # Group chunks by page
    pages = {}
    for chunk in chunks:
        page = chunk.get("page", 0)
        if page not in pages:
            pages[page] = []
        pages[page].append(chunk)

    total_pages = max(pages.keys()) if pages else 0

    # Print document header
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + f"  DOCUMENT: {doc_id}".ljust(68) + "║")
    print("║" + f"  Total Chunks: {len(chunks)}  |  Pages: {total_pages}".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Render each page
    for page_num in sorted(pages.keys()):
        page_chunks = pages[page_num]

        print(f"  ┌{'─' * 64}┐")
        print(f"  │{'PAGE ' + str(page_num):^64}│")
        print(f"  └{'─' * 64}┘")
        print()

        for chunk in page_chunks:
            content = clean_pua_chars(chunk.get("content", ""))
            content_type = chunk.get("content_type", "text")
            element_type = chunk.get("metadata", {}).get("element_type", "")

            if content_type == "table":
                # Render tables with a border
                print("  ┌─ TABLE ─────────────────────────────────────────────────────┐")
                for line in content.split("\n"):
                    line = clean_pua_chars(line)
                    # Truncate very long lines for terminal readability
                    if len(line) > 64:
                        print(f"  │ {line[:62]}…│")
                    else:
                        print(f"  │ {line.ljust(63)}│")
                print("  └─────────────────────────────────────────────────────────────┘")
                print()

            elif content_type == "image":
                print(f"  🖼️  [IMAGE OCR]: {content[:80]}{'…' if len(content) > 80 else ''}")
                print()

            elif element_type in ("SectionHeaderItem", "TitleItem"):
                # Render headers with emphasis
                print(f"  ── {content} ──")
                print()

            elif element_type == "ListItem":
                # Render list items with bullet points
                print(f"    • {content}")

            elif element_type == "CodeItem":
                print(f"    💻 {content}")

            elif element_type == "FormulaItem":
                print(f"    📐 {content}")

            else:
                # Regular text
                print(f"  {content}")

        print()
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_documents()
        print("Usage: python view_extracted.py <document_id>")
    else:
        render_document(sys.argv[1])
