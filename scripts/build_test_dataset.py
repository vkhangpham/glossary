import json
import shutil
from pathlib import Path

"""Build / refresh the synthetic test dataset used by sense_disambiguation unit tests.

Running this script is idempotent: it wipes the existing test_dataset directory and
re-creates the minimal hierarchy stub, empty unified-context file, and empty
raw_resources sub-tree for every selected term.

Actual resource harvesting is performed elsewhere; this script guarantees the
baseline folder layout so that other tools (scrapers, validators, tests) can
rely on consistent paths.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "sense_disambiguation" / "data" / "test_dataset"
RAW_ROOT = DATA_ROOT / "raw_resources"

POSITIVE_TERMS = [
    "transformers",
    "interface",
    "modeling",
    "fragmentation",
    "clustering",
    "stress",
    "regression",
    "cell",
    "network",
    "bond",
]
NEGATIVE_TERMS = [
    "artificial intelligence",
    "mathematics",
    "engineering",
    "geology",
    "astrophysics",
    "botany",
    "microbiology",
    "cryptography",
]
ALL_TERMS = POSITIVE_TERMS + NEGATIVE_TERMS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main build steps
# ---------------------------------------------------------------------------

def build_clean_tree() -> None:
    """Delete previous dataset (if any) and recreate folder hierarchy."""
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    (RAW_ROOT).mkdir(parents=True, exist_ok=True)

    # create per-term sub-dirs with .gitkeep placeholder
    for term in ALL_TERMS:
        term_dir = RAW_ROOT / term.replace(" ", "_")
        term_dir.mkdir(parents=True, exist_ok=True)
        (term_dir / ".gitkeep").touch()

    # hierarchy stub (identical to manually-crafted one in repo)
    hierarchy_stub_path = DATA_ROOT / "hierarchy.json"
    if not hierarchy_stub_path.exists():
        from json import loads as _loads
        # Import the stub directly from repo version if present
        stub_src = REPO_ROOT / "sense_disambiguation" / "data" / "test_dataset" / "hierarchy.json"
        if stub_src.exists():
            hierarchy_data = json.loads(stub_src.read_text())
        else:
            hierarchy_data = {
                "version": "0.1-test",
                "levels": {},
                "terms": {},
            }
        write_json(hierarchy_stub_path, hierarchy_data)

    # unified context stub
    write_json(DATA_ROOT / "unified_context_ground_truth.json", {"version": "1.0", "contexts": {}})


if __name__ == "__main__":
    build_clean_tree()
    print(f"Synthetic test dataset scaffolded under {DATA_ROOT.relative_to(REPO_ROOT)}") 