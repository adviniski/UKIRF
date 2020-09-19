import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.tools.functions import extract

if __name__ == "__main__":
    paths = [
    "../data/ml-100k-gte.csv"
    ]

    print(extract(paths))
    