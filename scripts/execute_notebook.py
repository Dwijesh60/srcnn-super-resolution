"""
Script to execute the generated Jupyter notebook top-to-bottom
and embed all outputs, plots, and tables into the .ipynb file.
"""

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "Image_Super_Resolution_SRCNN.ipynb"

def run_notebook():
    print(f"[*] Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    print(f"[*] Starting clean notebook execution across {len(nb.cells)} cells...")
    t0 = time.time()
    
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent.parent)}}
    )
    
    try:
        client.execute()
        print(f"[+] All cells executed successfully in {time.time() - t0:.2f}s!")
        
        # Save back the executed notebook with all outputs embedded
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print(f"[+] Executed notebook with rich outputs saved to: {NOTEBOOK_PATH}")
        return True
    except Exception as e:
        print(f"[!] Error during notebook execution: {e}")
        return False

if __name__ == "__main__":
    success = run_notebook()
    sys.exit(0 if success else 1)
