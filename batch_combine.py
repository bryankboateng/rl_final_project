import argparse
import pickle
import os

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def combine_batches(batch_files, index_ranges_per_file, output_file):
    combined = {}

    for batch_path, index_ranges in zip(batch_files, index_ranges_per_file):
        print(f"Loading: {batch_path}")
        data = load_pickle(batch_path)
        
        for key in data:
            if key not in combined:
                combined[key] = []

        for index_range in index_ranges:
            start, end = index_range
            for i in range(start, end + 1):
                for key in data:
                    combined[key].append(data[key][i])

    print(f"Saving combined batch to: {output_file}")
    save_pickle(combined, output_file)

def parse_index_ranges(raw):
    index_ranges = []
    for r in raw.split(","):
        if "-" in r:
            start, end = map(int, r.split("-"))
        else:
            start = end = int(r)
        index_ranges.append((start, end))
    return index_ranges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine batch.pkl files with selected index ranges.")

    parser.add_argument("--batch_files", nargs="+", required=True,
                        help="Paths to batch_*.pkl files.")
    parser.add_argument("--ranges", nargs="+", required=True,
                        help="Index ranges per file (e.g., 0-99 200-299). One string per file.")
    parser.add_argument("--output", required=True,
                        help="Output path for combined .pkl file.")

    args = parser.parse_args()

    assert len(args.batch_files) == len(args.ranges), "Must provide one --ranges string per --batch_files entry."

    index_ranges_per_file = [parse_index_ranges(r) for r in args.ranges]

    combine_batches(args.batch_files, index_ranges_per_file, args.output)
