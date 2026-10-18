from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen32-std4-20261017'))
from qwen32_model import ActualModel
