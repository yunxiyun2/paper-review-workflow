"""Entry point: `python main.py run configs/neurips_review.yaml --payload '...'`"""
import sys
from paper_review_workflow.cli import main

if __name__ == "__main__":
    sys.exit(main())
