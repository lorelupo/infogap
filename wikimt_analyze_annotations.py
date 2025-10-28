#!/usr/bin/env python3
"""
Analyze annotation files from the WikiGap InfoGap pipeline.

This script:
1. Loads annotation JSON files from a specified directory
2. Detects and optionally removes duplicate files (keeping most recent)
3. Computes symmetric reference divergence metrics for article pairs
4. Outputs summary statistics

Usage:
    python analyze_annotations.py
    python analyze_annotations.py --annotation-dir /path/to/annotations
    python analyze_annotations.py --delete-duplicates
    python analyze_annotations.py --min-facts 10
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_annotation_files(annotation_dir):
    """
    Load all annotation JSON files from the specified directory.
    
    Parameters:
    -----------
    annotation_dir : Path
        Directory containing annotation_*.json files
        
    Returns:
    --------
    list of pandas.DataFrame
        One DataFrame per annotation file
    """
    annotation_files = list(annotation_dir.glob("annotation_*.json"))
    
    if not annotation_files:
        print(f"⚠️  No annotation files found in {annotation_dir}")
        return []
    
    print(f"Found {len(annotation_files)} annotation files:")
    
    output_dfs = []
    
    for ann_file in annotation_files:
        print(f"  Loading {ann_file.name}...")
        
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Parse the Polars JSON structure
            columns_data = {}
            for col in data['columns']:
                col_name = col['name']
                values = col['values']
                
                # Handle nested list structures
                if col['datatype'] == {'List': 'Utf8'} or \
                   col['datatype'] == {'List': {'List': 'Utf8'}} or \
                   col['datatype'] == {'List': 'Float64'}:
                    # Keep as-is for nested structures
                    columns_data[col_name] = values
                else:
                    # Simple column
                    columns_data[col_name] = values
            
            # Create DataFrame - use the key columns we need
            df = pd.DataFrame({
                'fact': columns_data['fact'],
                'person_name': columns_data['person_name'],
                'language': columns_data['language'],
                'intersection_label': columns_data['intersection_label']
            })
            
            # Store filename for reference
            df.attrs['filename'] = ann_file.name
            
            output_dfs.append(df)
            print(f"    ✓ Loaded {len(df)} facts, languages: {df['language'].unique()}")
            
        except Exception as e:
            print(f"    ✗ Error loading {ann_file.name}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(output_dfs)} annotation files")
    return output_dfs


def detect_duplicates(annotation_dir, delete=False):
    """
    Detect and optionally remove duplicate annotation files.
    Keeps the most recent version of each article.
    
    Parameters:
    -----------
    annotation_dir : Path
        Directory containing annotation files
    delete : bool
        If True, delete older duplicate files
        
    Returns:
    --------
    dict
        Summary statistics about duplicates
    """
    annotation_files = list(annotation_dir.glob("annotation_*.json"))
    
    print(f"Found {len(annotation_files)} total annotation files")
    
    # Group files by article name (extract from filename)
    # Filename format: annotation_YYYY-MM-DD_ArticleName_lang.json
    article_files = defaultdict(list)
    
    for f in annotation_files:
        # Parse filename: annotation_2025-10-27_Article Name_fr.json
        # Strategy: split by '_', extract date (position 1), and reconstruct article name
        parts = f.stem.split('_')
        
        if len(parts) >= 3:
            # parts[0] = 'annotation'
            # parts[1] = date like '2025-10-27' (contains dashes)
            # parts[2:-1] = article name parts
            # parts[-1] = language code ('fr', 'en', etc.)
            
            date = parts[1]  # '2025-10-27'
            lang = parts[-1]  # 'fr' or 'en'
            
            # Article name is everything between date and language
            # Join parts[2:-1] with underscores
            article_name_parts = parts[2:-1]
            article_name = '_'.join(article_name_parts)
            
            # Use article name + language as unique key (same article may have both en and fr versions)
            unique_key = f"{article_name}_{lang}"
            
            article_files[unique_key].append((date, f))
    
    # Count unique articles (ignoring language suffix)
    unique_articles = set()
    for key in article_files.keys():
        # Remove _lang suffix to count unique article names
        article_base = '_'.join(key.split('_')[:-1]) if '_' in key else key
        unique_articles.add(article_base)
    
    print(f"Found {len(unique_articles)} unique articles ({len(article_files)} article-language pairs)")
    
    duplicates_found = 0
    files_to_remove = []
    
    print(f"\nArticles with duplicates:")
    
    for article_key, files in article_files.items():
        if len(files) > 1:
            duplicates_found += 1
            # Sort by date (descending) to keep the most recent
            files.sort(reverse=True)
            most_recent = files[0]
            older_files = files[1:]
            
            print(f"\n  {article_key}:")
            print(f"    ✓ Keeping:  {most_recent[1].name} ({most_recent[0]})")
            for date, f in older_files:
                print(f"    ✗ Removing: {f.name} ({date})")
                files_to_remove.append(f)
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total files: {len(annotation_files)}")
    print(f"  Unique articles: {len(unique_articles)}")
    print(f"  Article-language pairs: {len(article_files)}")
    print(f"  Pairs with duplicates: {duplicates_found}")
    print(f"  Files to remove: {len(files_to_remove)}")
    print(f"  Files after cleanup: {len(annotation_files) - len(files_to_remove)}")
    print(f"{'='*70}")
    
    # Delete if requested
    if files_to_remove and delete:
        print(f"\n⚠️  Deleting {len(files_to_remove)} older annotation files...")
        for f in files_to_remove:
            f.unlink()
            print(f"  Deleted: {f.name}")
        print(f"\n✓ Cleanup complete! Removed {len(files_to_remove)} files.")
    elif files_to_remove:
        print(f"\n⚠️  Found {len(files_to_remove)} duplicate files.")
        print("Run with --delete-duplicates to remove them.")
    else:
        print("\n✓ No duplicate files found!")
    
    return {
        'total_files': len(annotation_files),
        'unique_articles': len(article_files),
        'duplicates_found': duplicates_found,
        'files_removed': len(files_to_remove) if delete else 0
    }


def compute_reference_divergence(df, min_facts=6):
    """
    Compute symmetric reference divergence for an article pair.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'language', 'fact', 'intersection_label'
    min_facts : int
        Minimum total facts required (F_S + F_T) to compute divergence
        
    Returns:
    --------
    dict with keys:
        - 'divergence': float or None
        - 'F_S': int (source facts)
        - 'F_T': int (target facts)
        - 'Y_S': int (source facts matched in target)
        - 'Y_T': int (target facts matched in source)
        - 'excluded': bool (True if below threshold)
        - 'reason': str or None
    """
    
    # Identify source and target languages
    languages = df['language'].unique()
    if len(languages) != 2:
        return {
            'divergence': None,
            'F_S': 0, 'F_T': 0, 'Y_S': 0, 'Y_T': 0,
            'excluded': True,
            'reason': f'Expected 2 languages, found {len(languages)}'
        }
    
    src_lang, tgt_lang = languages[0], languages[1]
    
    # Get source and target subsets
    df_src = df[df['language'] == src_lang]
    df_tgt = df[df['language'] == tgt_lang]
    
    # Count facts
    F_S = len(df_src)
    F_T = len(df_tgt)
    
    # Check minimum threshold
    if F_S + F_T < min_facts:
        return {
            'divergence': None,
            'F_S': F_S, 'F_T': F_T, 'Y_S': 0, 'Y_T': 0,
            'excluded': True,
            'reason': f'Too few facts: F_S + F_T = {F_S + F_T} < {min_facts}'
        }
    
    # Count matches (yes labels from intersection_label column)
    def count_yes(labels_series):
        count = 0
        for label in labels_series:
            if isinstance(label, str):
                # Check for 'yes' in the label
                label_lower = label.lower().strip()
                if label_lower == 'yes' or label_lower == 'y':
                    count += 1
        return count
    
    Y_S = count_yes(df_src['intersection_label'])
    Y_T = count_yes(df_tgt['intersection_label'])
    
    # Compute divergence
    divergence = 1 - (Y_S + Y_T) / (F_S + F_T)
    
    return {
        'divergence': divergence,
        'F_S': F_S,
        'F_T': F_T,
        'Y_S': Y_S,
        'Y_T': Y_T,
        'excluded': False,
        'reason': None,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang
    }


def analyze_divergence(output_dfs, min_facts=6):
    """
    Compute reference divergence for all article pairs.
    
    Parameters:
    -----------
    output_dfs : list of pandas.DataFrame
        List of annotation DataFrames
    min_facts : int
        Minimum total facts threshold
        
    Returns:
    --------
    pandas.DataFrame
        Summary of divergence metrics for each article pair
    """
    divergence_results = []
    
    for i, df in enumerate(output_dfs):
        result = compute_reference_divergence(df, min_facts=min_facts)
        
        # Get article name if available
        if 'person_name' in df.columns:
            person_names = df['person_name'].unique()
            article_name = person_names[0] if len(person_names) > 0 else f'Article_{i}'
        else:
            article_name = f'Article_{i}'
        
        result['article'] = article_name
        result['filename'] = df.attrs.get('filename', f'unknown_{i}')
        divergence_results.append(result)
        
        # Print result
        if result['excluded']:
            print(f"⚠️  {article_name}: EXCLUDED - {result['reason']}")
        else:
            src_lang = result.get('src_lang', 'src')
            tgt_lang = result.get('tgt_lang', 'tgt')
            print(f"✓ {article_name} ({src_lang}↔{tgt_lang}):")
            print(f"  D_yes/no = {result['divergence']:.3f}")
            print(f"  Facts: F_{src_lang}={result['F_S']}, F_{tgt_lang}={result['F_T']}")
            print(f"  Matched: Y_{src_lang}={result['Y_S']}, Y_{tgt_lang}={result['Y_T']}")
            if result['F_S'] > 0 and result['F_T'] > 0:
                print(f"  Recall_{src_lang}={result['Y_S']/result['F_S']:.2%}, "
                      f"Recall_{tgt_lang}={result['Y_T']/result['F_T']:.2%}")
            print()
    
    # Create summary DataFrame
    divergence_df = pd.DataFrame(divergence_results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    display_cols = ['article', 'divergence', 'F_S', 'F_T', 'Y_S', 'Y_T', 'excluded']
    print(divergence_df[display_cols].to_string(index=False))
    
    non_excluded = divergence_df[~divergence_df['excluded']]
    if len(non_excluded) > 0:
        print(f"\nStatistics (non-excluded articles):")
        print(f"  Count: {len(non_excluded)}")
        print(f"  Mean divergence: {non_excluded['divergence'].mean():.3f}")
        print(f"  Median divergence: {non_excluded['divergence'].median():.3f}")
        print(f"  Min divergence: {non_excluded['divergence'].min():.3f}")
        print(f"  Max divergence: {non_excluded['divergence'].max():.3f}")
        print(f"  Std divergence: {non_excluded['divergence'].std():.3f}")
    else:
        print("\n⚠️  No valid article pairs to compute statistics")
    
    return divergence_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze annotation files from WikiGap InfoGap pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --annotation-dir /path/to/annotations
  %(prog)s --delete-duplicates
  %(prog)s --min-facts 10 --output results.csv
        """
    )
    
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='scratch/annotation_save/wikigap_data',
        help='Directory containing annotation JSON files (default: scratch/annotation_save/wikigap_data)'
    )
    
    parser.add_argument(
        '--delete-duplicates',
        action='store_true',
        help='Delete older duplicate annotation files (keeps most recent)'
    )
    
    parser.add_argument(
        '--min-facts',
        type=int,
        default=6,
        help='Minimum total facts (F_S + F_T) required to compute divergence (default: 6)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Optional: Save divergence results to CSV file'
    )
    
    parser.add_argument(
        '--skip-duplicates-check',
        action='store_true',
        help='Skip duplicate detection step'
    )
    
    args = parser.parse_args()
    
    # Resolve annotation directory
    annotation_dir = Path(args.annotation_dir)
    if not annotation_dir.is_absolute():
        annotation_dir = Path.cwd() / annotation_dir
    
    if not annotation_dir.exists():
        print(f"❌ Error: Annotation directory does not exist: {annotation_dir}")
        return 1
    
    print("="*70)
    print("WikiGap Annotation Analysis")
    print("="*70)
    print(f"Annotation directory: {annotation_dir}")
    print(f"Minimum facts threshold: {args.min_facts}")
    print("="*70)
    print()
    
    # Step 1: Detect and optionally remove duplicates
    if not args.skip_duplicates_check:
        print("STEP 1: Duplicate Detection")
        print("-"*70)
        dup_stats = detect_duplicates(annotation_dir, delete=args.delete_duplicates)
        print()
    
    # Step 2: Load annotation files
    print("STEP 2: Loading Annotations")
    print("-"*70)
    output_dfs = load_annotation_files(annotation_dir)
    
    if not output_dfs:
        print("❌ No annotation files to analyze")
        return 1
    
    print()
    
    # Step 3: Compute divergence metrics
    print("STEP 3: Computing Reference Divergence")
    print("-"*70)
    divergence_df = analyze_divergence(output_dfs, min_facts=args.min_facts)
    print()
    
    # Step 4: Save results if requested
    if args.output:
        output_path = Path(args.output)
        divergence_df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
        print()
    
    print("="*70)
    print("✓ Analysis complete!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
