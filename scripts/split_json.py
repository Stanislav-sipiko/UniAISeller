# /root/ukrsell_v4/scripts/split_json.py v1.0.0
import json
import os
from pathlib import Path

def split_products(input_path: str, chunk_size: int = 50):
    """
    Splits the products.json into chunks of 50 items for manual processing in Claude.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File {input_path} not found.")
        return

    # Create output directory
    output_dir = input_file.parent / "chunks"
    output_dir.mkdir(exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Split logic
    total = len(data)
    for i in range(0, total, chunk_size):
        chunk = data[i:i + chunk_size]
        part_num = (i // chunk_size) + 1
        output_file = output_dir / f"part_{part_num}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(chunk, f_out, ensure_ascii=False, indent=2)
        
        print(f"Saved: {output_file} ({len(chunk)} items)")

    print(f"\nDone. Total parts: {(total + chunk_size - 1) // chunk_size}")

if __name__ == "__main__":
    SOURCE = "/root/ukrsell_v4/stores/luckydog/products.json"
    split_products(SOURCE)