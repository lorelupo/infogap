import polars as pl
from packages.annotate import annotate_frame 
import ipdb

TARGET_LANG = "Korean"
TGT_LANG_CODE = "ko"
# Some packages you may need to pip install:
# - polars
# - loguru

# Instructions:
## You're going to be answering questions about whether a fact is present in a Wikipedia article.
## The two languages you'll be considering are English and Korean.
## For each fact, you'll be given the fact in one language (e.g., English)
##   and potentially relevant snippets from the Wikipedia article in the other language (e.g., Korean).
## If the English fact is conveyed by the Korean snippets, answer 'A'.
## If it is mostly conveyed by the Korean snippets, answer 'B'.
## If not, then open up the Korean Wikipedia article and see if the fact is present there. Then there are three possibilities:
    ## If the fact is present in the article, answer 'C'.
    ## If it is mostly (but not completely) present in the article, answer 'D'.
    ## If the fact is not present in the article, answer 'E'.
## It is up to your discretion to decide whether a fact is "mostly" present in the snippets or the article.

ANNOTATION_FNAME = "/Users/llupo/dev/infogap/scratch/annotation_save/annotation_2025-02-17_Sundae (sausage).json" # TODO: set to whatever directory you want to save the annotations to
frame = pl.read_json(ANNOTATION_FNAME)
en_link = "https://en.wikipedia.org/wiki/Sundae_(sausage)"
tgt_link = "https://ko.wikipedia.org/wiki/%EC%88%9C%EB%8C%80"

def ask_question(fact_row):
    src_lang = fact_row['language']
    # TODO: adjust this so that the context is a string.
    # convert to List[str] by making every inner list a string by enumerating its contents

    candidate_tgt_contexts = fact_row['tgt_contexts'] # List[List[str]]
    # candidate_tgt_contexts_translations = fact_row['tgt_contexts_en_translation']
    tgt_contexts_lst = [] # List[str]
    for i in range(len(candidate_tgt_contexts)):
        tgt_context = candidate_tgt_contexts[i]
        # tgt_context_translated = candidate_tgt_contexts_translations[i]
        # if src_lang == 'en':
        #     tgt_contexts_str = ''.join([f"{j+1}. {fact} ({tgt_context_translated[j]})\n" for j, fact in enumerate(tgt_context)]) 
        # else:
        tgt_contexts_str = ''.join([f"{j+1}. {fact}\n" for j, fact in enumerate(tgt_context)])
        tgt_contexts_lst.append(tgt_contexts_str)
    tgt_contexts_complete = '\n'.join(tgt_contexts_lst)
    tgt_lang = TARGET_LANG if fact_row['language'] == 'en' else 'English'

    # context_translated = fact_row['src_contexts_en_translation']
    context_str = fact_row['src_context']
    # if src_lang == 'fr':
    #     context_str = ''.join([f"{i+1}. {fact} ({context_translated[i]})\n" for i, fact in enumerate(context_str)])
    # else:
    context_str = ''.join([f"{i+1}. {fact}\n" for i, fact in enumerate(context_str)])

    link = en_link if src_lang == TGT_LANG_CODE else tgt_link
    # question = f"\n\nConsider the following fact(s) about {fact_row['person_name']}:\n\n{context_str}\n\nIs the final fact present in the {tgt_lang} Wikipedia article about {fact_row['person_name']})?\n\nHere are some snippets from the {tgt_lang} article:\n{tgt_contexts_complete}\n\n"
    question = f"\n\nConsider the following fact(s) about {fact_row['person_name']}:\n\n{context_str}\n\nIs the final fact present in the {tgt_lang} Wikipedia article about {fact_row['person_name']} ({link})?\n\nHere are some snippets from the {tgt_lang} article:\n{tgt_contexts_complete}\n\n"
    
    response_options = "\n".join(["A: covered by the snippets", "B: partly covered by the snippets", "C: covered by the article", "D: partly covered by the article", "E: Not in the article"])
    full_prompt = question + response_options + "\nAnswer (A/B/C/D/E): "
    return full_prompt
annotated_frame = annotate_frame(
    frame, # frame containing data to annotate 
    num_samples=20, # number of samples to annotate in one setting
    annotation_columns=['sam_annotations'], # column to store the annotation in
    question_fns=[ask_question],
    answer_validate_fn=[lambda answer: answer in ['A', 'B', 'C', 'D', 'E']]
)
annotated_frame.write_json(ANNOTATION_FNAME)