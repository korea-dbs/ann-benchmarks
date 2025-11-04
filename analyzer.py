#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import os


def analyze_result(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return False
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\n{'='*70}")
            print(f"File: {os.path.basename(filepath)}")
            print(f"{'='*70}\n")
            
            # Query Performance
            if 'times' in f:
                times = f['times'][:]
                print(f"Query Performance:")
                print(f"  {'─'*60}")
                print(f"  Total queries:      {len(times):,}")
                print(f"  QPS:                {1/np.mean(times):.2f} queries/sec")
                print(f"  Avg time:           {np.mean(times)*1000:.2f} ms")
                print(f"  Median time:        {np.median(times)*1000:.2f} ms")
                print(f"  Min time:           {np.min(times)*1000:.2f} ms")
                print(f"  Max time:           {np.max(times)*1000:.2f} ms")
                print(f"  Std dev:            {np.std(times)*1000:.2f} ms")
                print(f"  P95 time:           {np.percentile(times, 95)*1000:.2f} ms")
                print(f"  P99 time:           {np.percentile(times, 99)*1000:.2f} ms")
                print()
            else:
                print(f"No 'times' data found\n")
            
            # Recall (Accuracy)
            if 'recalls' in f:
                recalls = f['recalls'][:]
                print(f"Accuracy (Recall):")
                print(f"  {'─'*60}")
                print(f"  Mean recall:        {np.mean(recalls):.6f} ({np.mean(recalls)*100:.4f}%)")
                print(f"  Median recall:      {np.median(recalls):.6f} ({np.median(recalls)*100:.4f}%)")
                print(f"  Min recall:         {np.min(recalls):.6f}")
                print(f"  Max recall:         {np.max(recalls):.6f}")
                print()
            else:
                print(f"No 'recalls' data found\n")
            
            # Additional info
            if 'candidates' in f:
                candidates = f['candidates'][:]
                print(f"Search parameters:")
                print(f"  Candidates: {candidates}")
                print()
            
            # Attributes
            if f.attrs:
                print(f"Metadata:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                print()
            
            # All keys
            all_keys = list(f.keys())
            if len(all_keys) > 4:
                print(f"Available data: {', '.join(all_keys)}")
                print()
            
            print(f"{'='*70}\n")
            return True
            
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("ANN-Benchmarks Result Analyzer")
        print("="*70)
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} <path_to_hdf5_file>")
        print("\nExample:")
        print(f"  python3 {sys.argv[0]} results/fashion-mnist-784-euclidean/10/libsql-n32/euclidean_max_neighbors_32_use_index_true.hdf5")
        print()
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = analyze_result(filepath)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
