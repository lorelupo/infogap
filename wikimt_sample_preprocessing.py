#!/usr/bin/env python3
"""
Script preprocessing WikiMT data for the WikiGap pipeline.
Samples articles from treatment and control groups, retrieves their content,
cleans Wikipedia text, and generates the necessary configuration files for the InfoGap pipeline.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import pandas as pd
import pickle
import ijson
import dill
import importlib
import re
import json
from pathlib import Path
from collections import deque

from mwparserfromhell import parse, wikicode
from mwparserfromhell.parser import ParserError
from nltk.tokenize import sent_tokenize
from hanlp.utils.rules import split_sentence
import stanza


# --------------------------------------------------------------------------- #
# Command-line argument parsing
# --------------------------------------------------------------------------- #

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess WikiGap data: sample articles and generate config files'
    )
    
    # Required arguments
    parser.add_argument(
        '--tgt-lang',
        type=str,
        required=True,
        help='Target language code (e.g., fr, es, it)'
    )
    parser.add_argument(
        '--n-sample',
        type=int,
        required=True,
        help='Total number of articles to sample (must be even for equal split)'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--infogap-dir',
        type=str,
        default='/Users/llupo/dev/infogap',
        help='Path to infogap repository directory'
    )
    parser.add_argument(
        '--input-data-dir',
        type=str,
        default='/Users/llupo/dev/wiki_mt/data/',
        help='Path to wiki_mt data directory'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    
    # Validate N_sample is even
    if args.n_sample % 2 != 0:
        parser.error(f"--n-sample must be even (got {args.n_sample})")
    
    return args


# --------------------------------------------------------------------------- #
# Text cleaning functions
# --------------------------------------------------------------------------- #

# Global variable for Ukrainian NLP pipeline
_NLP_UK = None

def get_nlp_uk():
    """
    Lazily ensure models exist (without re-downloading) and return a cached pipeline.
    """
    global _NLP_UK
    if _NLP_UK is None:
        from stanza.resources.common import DownloadMethod
        try:
            stanza.download(
                'uk',
                processors='tokenize,mwt',
                verbose=False,
                download_method=DownloadMethod.REUSE_RESOURCES
            )
        except Exception:
            pass

        _NLP_UK = stanza.Pipeline(
            lang='uk',
            processors='tokenize,mwt',
            tokenize_no_ssplit=False,
            use_gpu=False,
            tokenize_pretokenized=False,
            download_method=DownloadMethod.REUSE_RESOURCES
        )
    return _NLP_UK


def load_auxiliary_sections(input_data_dir):
    """Load auxiliary section headers for different languages."""
    headers_dir = os.path.join(input_data_dir, "llm_header")
    langs = ['en', 'fr', 'zh', 'ru']
    
    AUXILIARY_SECTIONS = {}
    for lang in langs:
        path = Path(headers_dir) / f"{lang}_header.json"
        with path.open("r", encoding="utf-8") as f:
            AUXILIARY_SECTIONS[lang] = json.load(f)
    
    return AUXILIARY_SECTIONS


def compile_regex_patterns(AUXILIARY_SECTIONS):
    """Compile regex patterns for text cleaning."""
    LANG_MAP = {
        'en': 'english',
        'fr': 'french',
        'it': 'italian',
        'ru': 'russian',
        'zh': 'chinese',
        'zh-tw': 'chinese',
        'es': 'spanish',
        'uk': "ukrainian"
    }
    
    AUX_PATTERNS = {
        lang: [re.compile(r'\b' + re.escape(section) + r'\b', re.IGNORECASE)
               for section in sections]
        for lang, sections in AUXILIARY_SECTIONS.items()
    }
    
    patterns = {
        'TABLE_RE': re.compile(r'\{\|[\s\S]*?\n\|}', flags=re.DOTALL),
        'PARAGRAPH_RE': re.compile(r'\n\s*\n'),
        'REF_RE': re.compile(r'<ref[^>/]*?/?>.*?</ref\s*>', flags=re.DOTALL | re.IGNORECASE),
        'SELFREF_RE': re.compile(r'<ref[^>/]*/\s*>', flags=re.IGNORECASE),
        'LIST_LINE_RE': re.compile(r'^[ \t]*[*#;:][^:].*$', re.MULTILINE),
        'LINE_NON_WORD_RE': re.compile(r'^([^\w\s].*)$', re.MULTILINE),
        'LINE_WORD_COL_RE': re.compile(r'^\s*\w+[\|:].*?$', re.MULTILINE),
        'PAREN_COMMA_RE': re.compile(r'[\(（][\s　]*[,，][\s　]*[\)）]'),
        'PAREN_EMPTY_RE': re.compile(r'[\(（][\s　]*[\)）]'),
        'NLBREAK_RE': re.compile(r'\n{3,}'),
        'MULTISPACE_RE': re.compile(r' {2,}'),
        'INFOBOX_TABLE_RE': re.compile(r'\{\|\s*class=.*?infobox.*?\|\}', re.DOTALL | re.IGNORECASE),
        'MAPFRAME_RE': re.compile(r'<mapframe.*?</mapframe>', re.DOTALL | re.IGNORECASE),
        'GEOJSON_RE': re.compile(r'\{["\']?type["\']?\s*:\s*["\']Feature["\'].*?\}', re.DOTALL),
        'INFOBOX_RE': re.compile(r"\{\{[Ii]nfobox[\s\S]*?\}\}", flags=re.MULTILINE),
    }
    
    return LANG_MAP, AUX_PATTERNS, patterns


def split_article(text, language, PARAGRAPH_RE, LANG_MAP, into_sentences=False):
    """Split article into paragraphs or sentences."""
    paragraphs = PARAGRAPH_RE.split(text)
    paragraphs = [p.strip('\n') for p in paragraphs if p.strip()]

    if not into_sentences:
        return paragraphs

    sentences = []
    is_chinese = language in ('zh', 'zh-tw')
    is_uk = (language == "uk")
    nlp_uk = get_nlp_uk() if is_uk else None
    
    for p in paragraphs:
        if is_chinese:
            sentences.extend(split_sentence(p))
        elif is_uk:
            sents = [s.text for s in nlp_uk(p).sentences]
            sentences.extend(sents)
        else:
            nltk_lang = LANG_MAP.get(language, language)
            sentences.extend(sent_tokenize(p, language=nltk_lang))

    return [s.strip() for s in sentences if s.strip()]


def detect_headers(text):
    """Return True if text starts with a MediaWiki section header."""
    return bool(re.match(r'^={2,}\s*[^=].*?[^=]\s*={2,}', text.lstrip()))


def remove_infobox(wikitext, INFOBOX_RE, INFOBOX_TABLE_RE, MAPFRAME_RE, GEOJSON_RE):
    """Remove all top-level infobox templates from Wikitext."""
    try:
        code = parse(wikitext)
        removed_any = False
        for tmpl in code.filter_templates():
            if "infobox" in tmpl.name.strip_code().lower():
                code.remove(tmpl)
                removed_any = True

        if removed_any:
            text = str(code)
        else:
            text = INFOBOX_RE.sub("", wikitext)

    except (ParserError, ValueError):
        text = INFOBOX_RE.sub("", wikitext)

    text = INFOBOX_TABLE_RE.sub('', text)
    text = MAPFRAME_RE.sub('', text)
    text = GEOJSON_RE.sub('', text)
    return text


def remove_auxiliary_sections(text, language, AUX_PATTERNS, PARAGRAPH_RE, LANG_MAP, remove_headers=True):
    """Strip everything from the first auxiliary-section header onward."""
    try:
        aux_patterns = AUX_PATTERNS[language]
    except KeyError:
        raise ValueError(f"Language {language!r} not supported")

    segments = deque(split_article(text, language, PARAGRAPH_RE, LANG_MAP, into_sentences=False))
    result_segments = []

    while segments:
        segment = segments.popleft().lstrip()
        if not segment:
            continue

        while detect_headers(segment):
            header_line, *rest = segment.split('\n', 1)

            if any(p.search(header_line) for p in aux_patterns):
                return '\n\n'.join(result_segments)

            if not remove_headers:
                result_segments.append(header_line)

            segment = rest[0].lstrip() if rest else ''
            if not segment:
                break

        if segment:
            result_segments.append(segment)

    return '\n\n'.join(result_segments)


def clean_extracts(text, language, AUXILIARY_SECTIONS, LANG_MAP, AUX_PATTERNS, patterns, 
                   remove_headers=True, into_sentences=False):
    """
    Clean Wikipedia text and remove infobox remnants and markup elements.
    """
    if not text:
        return ""

    if language not in AUXILIARY_SECTIONS:
        raise ValueError(f"Language {language} not supported")
    
    # Kill all wikitables
    text = patterns['TABLE_RE'].sub('', text)
    
    # Remove infobox
    text = remove_infobox(text, patterns['INFOBOX_RE'], patterns['INFOBOX_TABLE_RE'], 
                         patterns['MAPFRAME_RE'], patterns['GEOJSON_RE'])

    # Remove references
    text = patterns['SELFREF_RE'].sub('', text)
    text = patterns['REF_RE'].sub('', text)

    # Remove all lists
    text = patterns['LIST_LINE_RE'].sub('', text)
    
    # Store header information BEFORE stripping MediaWiki markup
    header_info = {}
    
    if not remove_headers:
        lines_before = text.split('\n')
        for line in lines_before:
            line_stripped = line.strip()
            if detect_headers(line_stripped):
                match = re.match(r'^(={2,})\s*([^=].*?[^=])\s*\1', line_stripped)
                if match:
                    header_text = match.group(2).strip()
                    header_info[header_text.lower()] = line_stripped
    
    # Process paragraphs until we hit an auxiliary section
    text = remove_auxiliary_sections(text, language, AUX_PATTERNS, patterns['PARAGRAPH_RE'], 
                                    LANG_MAP, remove_headers=remove_headers)

    # Remove MediaWiki markup
    text = parse(text)
    text = wikicode.Wikicode.strip_code(text)

    # Additional regex cleaning
    text = patterns['LINE_NON_WORD_RE'].sub('', text)
    text = patterns['LINE_WORD_COL_RE'].sub('', text)
    text = patterns['PAREN_COMMA_RE'].sub(' ', text)
    text = patterns['PAREN_EMPTY_RE'].sub(' ', text)
    
    # Clean up whitespace
    text = patterns['NLBREAK_RE'].sub('\n\n', text)
    text = patterns['MULTISPACE_RE'].sub(' ', text)

    # Sentence segmentation if required
    if into_sentences:
        segments = split_article(text, language, patterns['PARAGRAPH_RE'], LANG_MAP, into_sentences=True)
        text = '\n'.join(segments)

    text = text.strip()

    # Restore header markers with ORIGINAL levels
    if not remove_headers and header_info:
        lines = text.split('\n')
        restored_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower() in header_info:
                original_header = header_info[line_stripped.lower()]
                restored_lines.append(original_header)
            else:
                restored_lines.append(line)
        text = '\n'.join(restored_lines)

    return text


# --------------------------------------------------------------------------- #
# Data loading and sampling functions
# --------------------------------------------------------------------------- #

def load_treatment_control_groups(input_data_langdir):
    """Load treatment and control group CSV files."""
    print("=" * 80)
    print("STEP 1: Loading treatment and control group data")
    print("=" * 80)
    
    treatment_file = os.path.join(
        input_data_langdir, 
        'treatment_group_rebalanced_after_sent_div_06B_d30.csv'
    )
    control_file = os.path.join(
        input_data_langdir, 
        'control_group_rebalanced_after_sent_div_06B_d30.csv'
    )
    
    print(f"Loading treatment group from: {treatment_file}")
    df_treat_group = pd.read_csv(treatment_file)
    print(f"  ✓ Loaded {len(df_treat_group)} treatment articles")
    
    print(f"Loading control group from: {control_file}")
    df_control_group = pd.read_csv(control_file)
    print(f"  ✓ Loaded {len(df_control_group)} control articles")
    
    return df_treat_group, df_control_group


def sample_articles(df_treat_group, df_control_group, n_sample, random_seed):
    """Sample articles from treatment and control groups."""
    print("\n" + "=" * 80)
    print(f"STEP 2: Sampling {n_sample} articles ({n_sample//2} treatment + {n_sample//2} control)")
    print("=" * 80)
    
    df_sampled_treat = df_treat_group.sample(n=n_sample//2, random_state=random_seed)
    df_sampled_treat_ids = set(df_sampled_treat['translationId'].tolist())
    print(f"  ✓ Sampled {len(df_sampled_treat_ids)} treatment articles")
    
    df_sampled_control = df_control_group.sample(n=n_sample//2, random_state=random_seed)
    df_sampled_control_ids = set(df_sampled_control['qid'].tolist())
    print(f"  ✓ Sampled {len(df_sampled_control_ids)} control articles")
    
    return df_sampled_treat, df_sampled_treat_ids, df_sampled_control, df_sampled_control_ids


def retrieve_revision_contents(input_data_langdir, df_sampled_treat_ids, df_sampled_control_ids):
    """Retrieve revision contents for sampled articles."""
    print("\n" + "=" * 80)
    print("STEP 3: Retrieving revision contents")
    print("=" * 80)
    
    json_path_treatment = os.path.join(input_data_langdir, 'revision_history_contents.json')
    json_path_control = os.path.join(input_data_langdir, 'revision_history_contents_control.json')
    id_field_treatment = 'translationId'
    id_field_control = 'qid'
    sample_content_treat = []
    sample_content_control = []
    
    for json_path, id_field, df_sampled_ids, sample_contents in [
        (json_path_treatment, id_field_treatment, df_sampled_treat_ids, sample_content_treat),
        (json_path_control, id_field_control, df_sampled_control_ids, sample_content_control)
    ]:
        print('-' * 80)
        print(f"Processing {json_path}...")
        print('-' * 80)
        
        with open(json_path, 'rb') as f:
            for article in ijson.items(f, 'item'):
                id_val = article[id_field]
                if id_val not in df_sampled_ids:
                    continue
                
                # Append to the list
                sample_contents.append({
                    id_field: id_val,
                    'sourceTitle': article['sourceTitle'],
                    'targetTitle': article['targetTitle'],
                    'sourceLanguage': article['sourceLanguage'],
                    'targetLanguage': article['targetLanguage'],
                    'raw_source': article['d30']['sourceContent'],
                    'raw_target': article['d30']['targetContent']
                })
        
        print(f"  ✓ Retrieved {len(sample_contents)} articles")
        if len(sample_contents) > 0:
            print(f"  Sample shape: {list(sample_contents[0].keys())}")
    
    return sample_content_treat, sample_content_control


# --------------------------------------------------------------------------- #
# Data merging and persistence functions
# --------------------------------------------------------------------------- #

def merge_and_save(df_sampled_treat, df_sampled_control, sample_content_treat, 
                   sample_content_control, scratch_dir, tgt_lang, n_sample):
    """Merge sampled metadata with revision contents and save to pickle."""
    print("\n" + "=" * 80)
    print("STEP 4: Merging and saving combined dataframe")
    print("=" * 80)
    
    # Merge treatment group - keep left columns on overlap
    df_treat_merged = df_sampled_treat.merge(
        pd.DataFrame(sample_content_treat),
        left_on='translationId',
        right_on='translationId',
        how='inner',
        suffixes=('', '_drop')
    )
    # Drop duplicate columns from right
    df_treat_merged = df_treat_merged[[c for c in df_treat_merged.columns if not c.endswith('_drop')]]
    
    # Merge control group - keep left columns on overlap
    df_control_merged = df_sampled_control.merge(
        pd.DataFrame(sample_content_control),
        left_on='qid',
        right_on='qid',
        how='inner',
        suffixes=('', '_drop')
    )
    # Drop duplicate columns from right
    df_control_merged = df_control_merged[[c for c in df_control_merged.columns if not c.endswith('_drop')]]
    
    # Add group labels
    df_treat_merged['group'] = 'treatment'
    df_control_merged['group'] = 'control'
    
    # Combine into single dataframe
    df_combined = pd.concat([df_treat_merged, df_control_merged], ignore_index=True)
    
    # Save combined dataframe to pickle
    os.makedirs(scratch_dir, exist_ok=True)
    combined_pkl_path = os.path.join(scratch_dir, f'sampled_combined_{tgt_lang}_n{n_sample}.pkl')
    
    with open(combined_pkl_path, 'wb') as f:
        pickle.dump(df_combined, f)
    
    print(f"✓ Saved combined dataframe: {combined_pkl_path}")
    print(f"  Shape: {df_combined.shape}")
    print(f"  Columns: {list(df_combined.columns)}")
    print(f"  Group distribution: {df_combined['group'].value_counts().to_dict()}")
    
    return df_combined


# --------------------------------------------------------------------------- #
# Configuration file generation
# --------------------------------------------------------------------------- #

def generate_config_files(df_combined, infogap_dir, tgt_lang):
    """Generate scraped_titles_{tgt}.py and wikigap_topics_scrape.py files."""
    print("\n" + "=" * 80)
    print("STEP 5: Generating configuration files")
    print("=" * 80)
    
    # Extract (sourceTitle, targetTitle) pairs from df_combined
    title_pairs = list(zip(df_combined['sourceTitle'], df_combined['targetTitle']))
    
    # Generate scraped_titles_{tgt}.py file
    output_file = os.path.join(infogap_dir, 'packages', f'scraped_titles_{tgt_lang}.py')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header comment
        f.write("# Auto-generated file containing (English, Target language) topic tuples\n\n")
        
        # Write the list variable
        f.write("en_tgt_title_pairs = [\n")
        
        # Write each tuple
        for en_title, tgt_title in title_pairs:
            # Escape single quotes in titles
            en_title_escaped = en_title.replace("'", "\\'")
            tgt_title_escaped = tgt_title.replace("'", "\\'")
            f.write(f"    ('{en_title_escaped}', '{tgt_title_escaped}'),\n")
        
        # Close the list
        f.write("] \n")
    
    print(f"✓ Generated: {output_file}")
    
    # Generate wikigap_topics_scrape.py file
    output_file = os.path.join(infogap_dir, 'wikigap_topics_scrape.py')
    
    # Extract English titles from sample data
    selected_topics = [en_title for en_title, _ in title_pairs]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the list variable
        f.write("selected_topics = [ \n")
        
        # Write each topic
        for topic in selected_topics:
            # Escape double quotes in titles
            topic_escaped = topic.replace('"', '\\"')
            f.write(f'"{topic_escaped}",\n')
        
        # Close the list
        f.write("]\n")
    
    print(f"✓ Generated: {output_file}")
    print(f"  Total topics: {len(selected_topics)}")


# --------------------------------------------------------------------------- #
# Text cleaning and block persistence
# --------------------------------------------------------------------------- #

def clean_and_persist_blocks(df_combined, infogap_dir, input_data_dir):
    """Clean article text and persist blocks for InfoGap pipeline."""
    print("\n" + "=" * 80)
    print("STEP 6: Cleaning article text and persisting blocks")
    print("=" * 80)
    
    # Setup paths and import required modules
    REPO_ROOT = Path(infogap_dir)
    SRC_DIR = REPO_ROOT / "src"
    
    # Add paths to sys.path
    for path in (str(SRC_DIR), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Change to repo root
    original_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    
    # Reload flowmason if already imported
    if "flowmason" in sys.modules:
        importlib.reload(sys.modules["flowmason"])
    
    # Import required functions
    from main_scrape_bios import process_wikipedia_text
    from packages.constants import BIO_SAVE_DIR
    
    # Load auxiliary sections and compile patterns
    AUXILIARY_SECTIONS = load_auxiliary_sections(input_data_dir)
    LANG_MAP, AUX_PATTERNS, patterns = compile_regex_patterns(AUXILIARY_SECTIONS)
    
    # Create BIO_SAVE_DIR
    os.makedirs(BIO_SAVE_DIR, exist_ok=True)
    print(f"Saving blocks to: {BIO_SAVE_DIR}")
    
    # Process each article
    total_articles = len(df_combined)
    for idx, row in df_combined.iterrows():
        src_title = row['sourceTitle']
        tgt_title = row['targetTitle']
        src_lang = row['sourceLanguage']
        tgt_lang = row['targetLanguage']
        
        print(f"\n[{idx+1}/{total_articles}] Processing: {src_title} ({src_lang}) ↔ {tgt_title} ({tgt_lang})")
        
        # Clean source text
        cleaned_src = clean_extracts(
            row['raw_source'], 
            src_lang, 
            AUXILIARY_SECTIONS, 
            LANG_MAP, 
            AUX_PATTERNS, 
            patterns,
            remove_headers=False, 
            into_sentences=False
        )
        
        # Clean target text
        cleaned_tgt = clean_extracts(
            row['raw_target'], 
            tgt_lang, 
            AUXILIARY_SECTIONS, 
            LANG_MAP, 
            AUX_PATTERNS, 
            patterns,
            remove_headers=False, 
            into_sentences=False
        )
        
        # Process and save source blocks
        src_blocks = process_wikipedia_text(cleaned_src, src_lang)
        src_file = os.path.join(BIO_SAVE_DIR, f"{src_title}_{src_lang}.pkl")
        with open(src_file, "wb") as f:
            dill.dump(src_blocks, f)
        print(f"  ✓ Saved {src_title}_{src_lang}.pkl ({len(src_blocks)} blocks)")
        
        # Process and save target blocks
        tgt_blocks = process_wikipedia_text(cleaned_tgt, tgt_lang)
        tgt_file = os.path.join(BIO_SAVE_DIR, f"{tgt_title}_{tgt_lang}.pkl")
        with open(tgt_file, "wb") as f:
            dill.dump(tgt_blocks, f)
        print(f"  ✓ Saved {tgt_title}_{tgt_lang}.pkl ({len(tgt_blocks)} blocks)")
    
    # Restore original working directory
    os.chdir(original_cwd)
    
    print(f"\n✓ Processed and saved {total_articles * 2} article files")


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #

def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("WikiGap Preprocessing Script")
    print("=" * 80)
    print(f"Target language: {args.tgt_lang}")
    print(f"Sample size: {args.n_sample} ({args.n_sample//2} treatment + {args.n_sample//2} control)")
    print(f"Infogap directory: {args.infogap_dir}")
    print(f"Input data directory: {args.input_data_dir}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)
    
    # Construct paths
    input_data_langdir = os.path.join(
        args.input_data_dir, 
        f'parallel_corpora_enriched_split/en/{args.tgt_lang}/'
    )
    scratch_dir = os.path.join(args.infogap_dir, 'scratch')
    
    # Execute preprocessing steps
    df_treat_group, df_control_group = load_treatment_control_groups(input_data_langdir)
    
    df_sampled_treat, df_sampled_treat_ids, df_sampled_control, df_sampled_control_ids = sample_articles(
        df_treat_group, df_control_group, args.n_sample, args.random_seed
    )
    
    sample_content_treat, sample_content_control = retrieve_revision_contents(
        input_data_langdir, df_sampled_treat_ids, df_sampled_control_ids
    )
    
    df_combined = merge_and_save(
        df_sampled_treat, df_sampled_control, sample_content_treat, 
        sample_content_control, scratch_dir, args.tgt_lang, args.n_sample
    )
    
    generate_config_files(df_combined, args.infogap_dir, args.tgt_lang)
    
    # Clean and persist blocks
    clean_and_persist_blocks(df_combined, args.infogap_dir, args.input_data_dir)
    
    print("\n" + "=" * 80)
    print("✓ Preprocessing completed successfully!")
    print("=" * 80)
    
    print(f"\nNext steps:")
    print(f"1. Run main_complete_analysis.py to execute the InfoGap pipeline")
    print(f"   Example: python main_complete_analysis.py run-multiple-topics --model=gpt-4o-mini")


if __name__ == '__main__':
    main()
