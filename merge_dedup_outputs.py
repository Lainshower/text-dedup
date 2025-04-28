import os
import glob
import argparse


def merge_language_outputs(output_dir):
    merged_count = 0
    for lang_dir in sorted(os.listdir(output_dir)):
        lang_path = os.path.join(output_dir, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        dedup_files = sorted(glob.glob(os.path.join(lang_path, 'dedup_node*.jsonl')))
        if not dedup_files:
            continue
        out_path = os.path.join(output_dir, f"{lang_dir}.jsonl")
        print(f"Merging {len(dedup_files)} files for language '{lang_dir}' into {out_path}")
        with open(out_path, 'w', encoding='utf-8') as fout:
            for dedup_file in dedup_files:
                with open(dedup_file, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
        merged_count += 1
    print(f"Merged {merged_count} languages.")


def main():
    parser = argparse.ArgumentParser(description="Merge dedup_node*.jsonl files per language into a single <lang>.jsonl file.")
    parser.add_argument('output_dir', type=str, help='Path to the dedup output directory (containing language subdirs)')
    args = parser.parse_args()
    merge_language_outputs(args.output_dir)

if __name__ == "__main__":
    main() 