from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.repair.g03_q023_repair import merge_g03_q023_repair


if __name__ == "__main__":
    print(merge_g03_q023_repair())
