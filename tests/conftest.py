import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# macOS writes AppleDouble sidecars (._*) next to every file on this exFAT
# drive; they are binary resource forks, not importable modules.
collect_ignore_glob = ["._*"]
