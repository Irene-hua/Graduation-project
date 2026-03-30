#!/usr/bin/env python3
"""
Inspect Qdrant collection contents for debugging ingestion behavior.

Usage:
  python scripts/inspect_collection.py --config config/config.yaml [--file data/single_test1/test1.txt]

This script prints:
- collection name and point count
- top N source files and counts
- whether a specific file is present

This helps verify that ingestion appends points rather than overwriting.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import yaml
import argparse
from collections import Counter

from src.retrieval import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--collection_name', type=str, default=None, help='Override Qdrant collection name to inspect')
    parser.add_argument('--file', type=str, default=None, help='Optional source file path to look for in payloads')
    parser.add_argument('--hash', type=str, default=None, help='Optional content_hash (sha256) to look for')
    parser.add_argument('--limit', type=int, default=10000, help='Max records to scroll')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    collection_name = args.collection_name or config['vector_db']['collection_name']
    vs = VectorStore(
        collection_name=collection_name,
        dimension=config['embedding'].get('dimension', 384),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path'],
        host=config['vector_db'].get('host'),
        port=config['vector_db'].get('port')
    )

    try:
        print('Collection:', vs.collection_name)
        try:
            info = vs.get_collection_info()
            total = info.get('points_count')
        except Exception:
            info = {}
            total = None
        print('Collection info:', info)
        print('Total points (from get_collection_info/count):', total)

        # Scroll payloads (local qdrant supports scroll)
        try:
            records = vs.client.scroll(collection_name=vs.collection_name, limit=args.limit, with_payload=True, with_vectors=False)
            recs = records[0] if isinstance(records, tuple) else records
        except Exception as e:
            print('Failed to scroll from client:', e)
            recs = []

        src_counter = Counter()
        doc_id_counter = Counter()
        found_file = False
        found_hash = False
        sample_payloads = []

        for rec in recs:
            payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
            try:
                # Use VectorStore's payload normalization helper
                pd = vs._normalize_payload(payload)
            except Exception:
                pd = {}
            # prefer new source_path if available
            src = pd.get('source_path') or pd.get('source_file') or pd.get('doc_filepath') or pd.get('filepath') or pd.get('filename')
            if src:
                src_counter[src] += 1
            doc_id = pd.get('doc_id')
            if doc_id is not None:
                doc_id_counter[doc_id] += 1

            # check for specific file
            if args.file:
                # normalize both sides
                try:
                    norm_src = os.path.normpath(src) if src else None
                    norm_query = os.path.normpath(args.file)
                except Exception:
                    norm_src = src
                    norm_query = args.file

                if src and (norm_src == norm_query or os.path.basename(norm_src) == os.path.basename(norm_query) or norm_src.endswith(norm_query) or norm_src.endswith(os.path.basename(norm_query))):
                    found_file = True
                    sample_payloads.append(pd)

            # check for content hash
            if args.hash:
                ch = pd.get('content_hash')
                if ch and ch == args.hash:
                    found_hash = True
                    sample_payloads.append(pd)

        print('\nTop source files (by stored chunks):')
        for src, cnt in src_counter.most_common(20):
            print(f'  {src}: {cnt}')

        print('\nDistinct doc_id counts (sample):')
        for did, cnt in doc_id_counter.most_common(20):
            print(f'  {did}: {cnt}')

        print('\nInterpretation: if two datasets are both present, you should see both source paths/hashes here. If only one appears after repeated imports, use --reset_collection for a clean import or --collection_name to isolate corpora.')

        if args.file:
            print(f"\nLooking for file: {args.file}")
            print('Found in collection:' , found_file)
            if found_file:
                print('\nSample payloads for this file (first 3):')
                for p in sample_payloads[:3]:
                    # show new provenance fields clearly
                    display = {k: (v if len(str(v))<200 else str(v)[:200]+'...') for k,v in p.items()}
                    print('  ', display)

        if args.hash:
            print(f"\nLooking for content_hash: {args.hash}")
            print('Found by hash in collection:' , found_hash)
            if found_hash and not args.file:
                print('\nSample payloads for this hash (first 3):')
                for p in sample_payloads[:3]:
                    display = {k: (v if len(str(v))<200 else str(v)[:200]+'...') for k,v in p.items()}
                    print('  ', display)

    finally:
        # explicit close
        try:
            if hasattr(vs, 'client') and vs.client is not None:
                vs.client.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
