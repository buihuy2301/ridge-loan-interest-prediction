import sys
from pathlib import Path

# Let `import src.<module>` work when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).parent))
