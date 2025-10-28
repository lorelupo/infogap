import pandas as pd
import polars as pl
import ipdb
from flowmason import SingletonStep
from packages.info_diff_caa import step_forced_align_facts_to_paragraph, step_forced_align_en_tgt_facts_to_paragraph 
from packages.steps.caa_steps import step_prep_for_caa, step_caa_multi_sentence, step_infer_pronoun, step_prep_for_caa_en_tgt, step_caa_multi_sentence_en_tgt, step_caa_multi_sentence_flan
from packages.constants import EVENT_SAVE_DIR, TGT_LANG
from typing import Optional, Dict, List
from functools import partial
from collections import OrderedDict

from packages.steps.info_diff_steps import step_generate_facts, step_obtain_paragraphs_associations,\
      step_obtain_en_zh_paragraphs_associations, step_retrieve_potential_matches, step_compute_info_gap_reasoning, step_union_alignments, step_collapse_gpt_labels,\
      step_retrieve_prescraped_en_content_blocks, step_retrieve_prescraped_fr_content_blocks, step_retrieve_prescraped_zh_content_blocks,\
      step_retrieve_prescraped_tgt_content_blocks, step_compute_info_gap_reasoning_flan,\
      step_retrieve_prescraped_ru_content_blocks, step_obtain_en_ru_paragraphs_associations, step_obtain_en_tgt_paragraphs_associations, \
      step_retrieve_prescraped_ru_content_blocks, step_retrieve_potential_matches_en_tgt,\
      step_extract_person_abln, step_get_tgt_retrieval_candidates,\
      step_compute_info_gap_reasoning_simple

from packages.steps.caa_steps import step_caa_multi_sentence, step_caa_multi_sentence_flan 

def get_en_fr_info_diff_map_dict(en_bio_id=None, fr_bio_id=None, person_name=None):
    map_reduce_dict = OrderedDict()

    ## English
    en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    fr_bio_id_dict = {'fr_bio_id': fr_bio_id} if fr_bio_id else {}
    person_name_dict = {'person_name': person_name} if person_name else {}
    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
        'version': '003', 
        **en_bio_id_dict
    },)
    map_reduce_dict['step_get_fr_content_blocks'] = SingletonStep(step_retrieve_prescraped_fr_content_blocks, { # in info diff steps
        'version': '003', 
        **fr_bio_id_dict
    })
    ## Repeat, but for French
    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '003',
        'lang_code': 'en',
        'content_blocks': 'step_get_en_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_generate_facts_fr'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '002',
        'lang_code': 'fr', 
        'content_blocks': 'step_get_fr_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_infer_pronoun'] = SingletonStep(step_infer_pronoun, {
        'version': '001',
        'en_content_blocks': 'step_get_en_content_blocks' 
    })
    #### 
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'fr_facts': 'step_generate_facts_fr'
    })
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs'
    })

    map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches, {
        'version': '010',
        'en_facts': 'step_generate_facts',
        'fr_facts': 'step_generate_facts_fr',
        'alignment_df': 'step_union_fact_paragraphs', 
        **en_bio_id_dict,
        **fr_bio_id_dict,
        **person_name_dict
    })
    ### 

    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': '006',
        'model_name': 'gpt-4',
        'info_gap_retrieval_dfs': 'step_find_retrieval_candidates',
        **person_name_dict
    })
    map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
        'version': '002',
        'model_intersection_names': ('gpt-4',), 
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    })
    # TODO: this has to be updated since we pass content blocks (possibly containing headers) rather than paragraphs
    map_reduce_dict['step_add_fact_to_sent_alignment_info'] = SingletonStep(step_forced_align_facts_to_paragraph, {
        'version': '002',
        'en_fr_info_gaps': 'step_collapse_gpt_labels',
        'en_content_blocks': 'step_get_en_content_blocks',
        'fr_content_blocks': 'step_get_fr_content_blocks',
        'pronoun': 'step_infer_pronoun'
    })
    return map_reduce_dict


# def get_en_zh_info_diff_map_dict(en_bio_id=None, zh_bio_id=None, person_name=None, zh_person_name=None):
#     map_reduce_dict = OrderedDict()

#     ## English
#     en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
#     zh_bio_id_dict = {'tgt_bio_id': zh_bio_id} if zh_bio_id else {}
#     person_name_dict = {'person_name': person_name, 'tgt_person_name': zh_person_name} if person_name else {}
#     print(zh_bio_id_dict)
#     map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
#         'version': '003', 
#         **en_bio_id_dict
#     },)
#     map_reduce_dict['step_get_zh_content_blocks'] = SingletonStep(step_retrieve_prescraped_zh_content_blocks, { # in info diff steps
#         'version': '003', 
#         **zh_bio_id_dict
#     })
#     ## Repeat, but for chinese
#     map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
#         'version': '003',
#         'lang_code': 'en',
#         'content_blocks': 'step_get_en_content_blocks', 
#         **person_name_dict
#     })
#     map_reduce_dict['step_generate_facts_zh'] = SingletonStep(step_generate_facts, { # in info diff steps
#         'version': '002',
#         'lang_code': 'zh', 
#         'content_blocks': 'step_get_zh_content_blocks', 
#         **person_name_dict
#     })
#     # map_reduce_dict['step_infer_pronoun'] = SingletonStep(step_infer_pronoun, {
#     #     'version': '001',
#     #     'en_content_blocks': 'step_get_en_content_blocks' 
#     # })
#     #### 
#     map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_en_zh_paragraphs_associations, {
#         'version': '003',
#         'en_facts': 'step_generate_facts',
#         'zh_facts': 'step_generate_facts_zh'
#     })
#     map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
#         'version': '002',
#         'unpruned_alignment_strns': 'step_align_fact_paragraphs',
#         'lang_code': 'zh'
#     })

#     map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches_en_tgt, {
#         'version': '010',
#         'en_facts': 'step_generate_facts',
#         'tgt_facts': 'step_generate_facts_zh',
#         'alignment_df': 'step_union_fact_paragraphs', 
#         'lang_code': 'zh',
#         **en_bio_id_dict,
#         **zh_bio_id_dict,
#         **person_name_dict,
#         'tgt_bio_id': zh_bio_id
#     })
#     ### 

#     map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
#         'version': '006',
#         'model_name': 'gpt-5-mini',
#         'lang_code': 'zh',
#         'info_gap_retrieval_dfs': 'step_find_retrieval_candidates',
#         **person_name_dict
#     })
#     map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
#         'version': '002',
#         'model_intersection_names': ('gpt-5-mini',), 
#         'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
#     })
#     # TODO: this has to be updated since we pass content blocks (possibly containing headers) rather than paragraphs
#     map_reduce_dict['step_add_fact_to_sent_alignment_info'] = SingletonStep(step_forced_align_en_tgt_facts_to_paragraph, {
#         'version': '002',
#         'en_tgt_info_gaps': 'step_collapse_gpt_labels',
#         'en_content_blocks': 'step_get_en_content_blocks',
#         'tgt_content_blocks': 'step_get_zh_content_blocks',
#         'pronoun': None
#     })
#     return map_reduce_dict


def get_en_zh_info_diff_map_dict(en_bio_id=None, zh_bio_id=None, person_name=None, tgt_person_name=None):
    map_reduce_dict = OrderedDict()

    ## English
    en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    zh_bio_id_dict = {'tgt_bio_id': zh_bio_id} if zh_bio_id else {}
    person_name_dict = {'person_name': person_name, 'tgt_person_name': tgt_person_name} if person_name else {}
    print(zh_bio_id_dict)
    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
        'version': '003', 
        **en_bio_id_dict
    },)
    map_reduce_dict['step_get_zh_content_blocks'] = SingletonStep(step_retrieve_prescraped_zh_content_blocks, { # in info diff steps
        'version': '003', 
        **zh_bio_id_dict
    })
    ## Repeat, but for chinese
    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '003',
        'lang_code': 'en',
        'content_blocks': 'step_get_en_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_generate_facts_zh'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '002',
        'lang_code': 'zh', 
        'content_blocks': 'step_get_zh_content_blocks', 
        **person_name_dict
    })
    # map_reduce_dict['step_infer_pronoun'] = SingletonStep(step_infer_pronoun, {
    #     'version': '001',
    #     'en_content_blocks': 'step_get_en_content_blocks' 
    # })
    #### 
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_en_zh_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'zh_facts': 'step_generate_facts_zh'
    })
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs',
        'lang_code': 'zh'
    })

    map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches_en_tgt, {
        'version': '010',
        'en_facts': 'step_generate_facts',
        'tgt_facts': 'step_generate_facts_zh',
        'alignment_df': 'step_union_fact_paragraphs', 
        'lang_code': 'zh',
        **en_bio_id_dict,
        **zh_bio_id_dict,
        **person_name_dict,
        'tgt_bio_id': zh_bio_id
    })
    ### 

    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': '006',
        'model_name': 'gpt-5-mini',
        'lang_code': 'zh',
        'info_gap_retrieval_dfs': 'step_find_retrieval_candidates',
        **person_name_dict
    })
    map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
        'version': '002',
        'model_intersection_names': ('gpt-5-mini',), 
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    })
    # TODO: this has to be updated since we pass content blocks (possibly containing headers) rather than paragraphs
    map_reduce_dict['step_add_fact_to_sent_alignment_info'] = SingletonStep(step_forced_align_en_tgt_facts_to_paragraph, {
        'version': '002',
        'en_tgt_info_gaps': 'step_collapse_gpt_labels',
        'en_content_blocks': 'step_get_en_content_blocks',
        'tgt_content_blocks': 'step_get_zh_content_blocks',
        'pronoun': None
    })
    return map_reduce_dict




def get_en_tgt_info_diff_map_dict(en_bio_id=None, tgt_bio_id=None, person_name=None, tgt_person_name=None, tgt_lang=None, model_name: str = 'gpt-5-mini', use_batch: bool = False, batch_poll_interval: int = 30, batch_timeout: int = 3600):
    map_reduce_dict = OrderedDict()

    ## English
    en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    tgt_bio_id_dict = {'tgt_bio_id': tgt_bio_id} if tgt_bio_id else {}
    person_name_dict = {'person_name': person_name, 'tgt_person_name': tgt_person_name} if person_name else {}
    tgt_lang_dict = {'tgt_lang': tgt_lang} if tgt_lang else {}
    tgt_lang = TGT_LANG
    print("targeted language: ", tgt_lang)

    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
        'version': '003', 
        **en_bio_id_dict
    },)
    map_reduce_dict['step_get_tgt_content_blocks'] = SingletonStep(step_retrieve_prescraped_tgt_content_blocks, { # in info diff steps
        'version': '003', 
        **tgt_bio_id_dict,
        **tgt_lang_dict
    })
    fact_version_en = '003-batch' if use_batch else '003'
    fact_version_tgt = '002-batch' if use_batch else '002'
    reasoning_version = '006-batch' if use_batch else '006'

    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': fact_version_en,
        'lang_code': 'en',
        'model_name': model_name,
        'content_blocks': 'step_get_en_content_blocks', 
        'use_batch': use_batch,
        'batch_poll_interval': batch_poll_interval,
        'batch_timeout': batch_timeout,
        **person_name_dict
    })
    map_reduce_dict['step_generate_facts_tgt'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': fact_version_tgt,
        'lang_code': tgt_lang,
        'model_name': model_name,
        'content_blocks': 'step_get_tgt_content_blocks', 
        'use_batch': use_batch,
        'batch_poll_interval': batch_poll_interval,
        'batch_timeout': batch_timeout,
        **person_name_dict
    })
    # Paragraph Alignment, correcting for Hubness between paragraphs
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_en_tgt_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'tgt_facts': 'step_generate_facts_tgt'
    })
    # Generate the Adjacency matrix
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs',
        'lang_code': tgt_lang,
    })
    # alighing facts based on the adjacency matrix, with correction for Hubness between facts
    map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches_en_tgt, {
        'version': '010',
        'en_facts': 'step_generate_facts',
        'tgt_facts': 'step_generate_facts_tgt',
        'alignment_df': 'step_union_fact_paragraphs', 
        'lang_code': tgt_lang,
        **en_bio_id_dict,
        **tgt_bio_id_dict,
        **person_name_dict,
        'tgt_bio_id': tgt_bio_id
    })
    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': reasoning_version,
        'model_name': model_name,
        'lang_code': tgt_lang,
        'use_batch': use_batch,
        'batch_poll_interval': batch_poll_interval,
        'batch_timeout': batch_timeout,
        'info_gap_retrieval_dfs': 'step_find_retrieval_candidates',
        **person_name_dict
    })
    map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
        'version': '002',
        'model_intersection_names': (model_name,), 
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    })
    # TODO: this has to be updated since we pass content blocks (possibly containing headers) rather than paragraphs
    map_reduce_dict['step_add_fact_to_sent_alignment_info'] = SingletonStep(step_forced_align_en_tgt_facts_to_paragraph, {
        'version': '002',
        'en_tgt_info_gaps': 'step_collapse_gpt_labels',
        'en_content_blocks': 'step_get_en_content_blocks',
        'tgt_content_blocks': 'step_get_tgt_content_blocks',
        'pronoun': None
    })
    return map_reduce_dict











def get_en_fr_ablation_dict() -> Dict:
    map_reduce_dict = OrderedDict()
    # en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    # fr_bio_id_dict = {'fr_bio_id': fr_bio_id} if fr_bio_id else {}
    map_reduce_dict['step_extract_annotation_frame_subset'] = SingletonStep(step_extract_person_abln, {
        'version': '001'
    })
    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, {
        'version': '001'
    })
    map_reduce_dict['step_get_fr_content_blocks'] = SingletonStep(step_retrieve_prescraped_fr_content_blocks, {
        'version': '001'
    })
    map_reduce_dict['step_get_en_facts_for_person'] = SingletonStep(step_generate_facts, {
        'version': '003',
        'lang_code': 'en',
        'content_blocks': 'step_get_en_content_blocks'
    })
    map_reduce_dict['step_get_fr_facts_for_person'] = SingletonStep(step_generate_facts, {
        'version': '004',
        'lang_code': 'fr',
        'content_blocks': 'step_get_fr_content_blocks'
    })
    map_reduce_dict['step_get_retrieval_candidates'] = SingletonStep(step_get_tgt_retrieval_candidates, { # TODO: have to implement a new function
        'version': '002',
        'all_en_fact_blocks': 'step_get_en_facts_for_person',
        'all_tgt_fact_blocks': 'step_get_fr_facts_for_person',
        'annotation_frame': 'step_extract_annotation_frame_subset',
    })
    map_reduce_dict['step_get_infogap_labels'] = SingletonStep(step_compute_info_gap_reasoning_simple, {
        'version': '001',
        'query_df': 'step_get_retrieval_candidates'
    })
    return map_reduce_dict

def get_en_ru_gpt_info_diff_map_dict(en_bio_id=None, ru_bio_id=None, person_name=None, 
                                     ru_person_name=None):
    map_reduce_dict = OrderedDict()
    en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    ru_bio_id_dict = {'tgt_bio_id': ru_bio_id} if ru_bio_id else {}
    person_name_dict = {'person_name': person_name, 'tgt_person_name': ru_person_name} if person_name else {}

    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
        'version': '003', 
        **en_bio_id_dict
    })
    map_reduce_dict['step_get_ru_content_blocks'] = SingletonStep(step_retrieve_prescraped_ru_content_blocks, { # in info diff steps
        'version': '003', 
        **ru_bio_id_dict
    })
    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '003',
        'lang_code': 'en',
        'content_blocks': 'step_get_en_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_generate_facts_ru'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '002',
        'lang_code': 'ru', 
        'content_blocks': 'step_get_ru_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_en_ru_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'ru_facts': 'step_generate_facts_ru'
    })
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs',
        'lang_code': 'ru'
    })
    map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches_en_tgt, {
        'version': '010',
        'en_facts': 'step_generate_facts',
        'tgt_facts': 'step_generate_facts_ru',
        'alignment_df': 'step_union_fact_paragraphs', 
        'lang_code': 'ru', # 'en' or 'ru
        **en_bio_id_dict,
        **ru_bio_id_dict,
        **person_name_dict
    })
    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': '007',
        'model_name': 'gpt-4',
        'lang_code': 'ru',
        'info_gap_retrieval_dfs': 'step_find_retrieval_candidates',
        **person_name_dict
    })
    map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
        'version': '002',
        'model_intersection_names': ('gpt-4', ),
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    })
    map_reduce_dict['step_infer_pronoun'] = SingletonStep(step_infer_pronoun, {
        'version': '001',
        'en_content_blocks': 'step_get_en_content_blocks' 
    })
    map_reduce_dict['step_add_fact_to_sent_alignment_info'] = SingletonStep(step_forced_align_en_tgt_facts_to_paragraph, {
        'version': '002',
        'en_tgt_info_gaps': 'step_collapse_gpt_labels',
        'en_content_blocks': 'step_get_en_content_blocks',
        'tgt_content_blocks': 'step_get_ru_content_blocks', 
        'pronoun': 'step_infer_pronoun'
    })
    return map_reduce_dict


def get_caa_map_dict_gpt():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_prep_for_caa'] = SingletonStep(step_prep_for_caa_en_tgt, {
        'version': '001', 
        'intersection_column': 'gpt-4o_intersection_label',
        'tgt_lang_code': 'ru'
    })
    map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_en_tgt, {
        'version': '001', 
        'en_tgt_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
        'tgt_lang_code': 'ru'
    })
    return map_reduce_dict

def get_caa_map_dict_gpt():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_prep_for_caa'] = SingletonStep(step_prep_for_caa_en_tgt, {
        'version': '001', 
        'intersection_column': 'gpt-4o_intersection_label',
        'tgt_lang_code': 'ru'
    })
    map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_en_tgt, {
        'version': '001', 
        'en_tgt_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
        'tgt_lang_code': 'ru'
    })
    return map_reduce_dict

def get_caa_map_dict_flan_ru():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_prep_for_caa'] = SingletonStep(step_prep_for_caa_en_tgt, {
        'version': '001', 
        'intersection_column': 'mt5-large_intersection_label',
        'tgt_lang_code': 'ru'
    })
    map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_flan, {
        'version': '002', 
        'en_other_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
        'other_lang_code': 'ru'
    })
    return map_reduce_dict

def get_caa_map_dict_fr():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_prep_for_caa'] = SingletonStep(step_prep_for_caa, {
        'version': '001', 
    })
    map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_flan, {
        'version': '001', 
        'en_fr_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
    })
    # map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_flan, {
    #     'version': '001', 
    #     'en_fr_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
    # })
    return map_reduce_dict

def get_caa_map_dict_fr_gpt():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_prep_for_caa'] = SingletonStep(step_prep_for_caa, {
        'version': '001', 
        'intersection_column': 'gpt-4o_intersection_label',
        'tgt_lang_code': 'fr'
    })
    map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_en_tgt, {
        'version': '001', 
        'en_tgt_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
        'tgt_lang_code': 'fr'
    })
    # map_reduce_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence_flan, {
    #     'version': '001', 
    #     'en_fr_info_gaps': 'step_prep_for_caa', # this should override the former one, i think
    # })
    return map_reduce_dict

def general_event_en_fr_map_dict():
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_en_content_blocks, { # in info diff steps
        'save_dir': EVENT_SAVE_DIR,
        'version': '003'
    })
    map_reduce_dict['step_get_fr_content_blocks'] = SingletonStep(step_retrieve_prescraped_fr_content_blocks, { # in info diff steps
        'save_dir': EVENT_SAVE_DIR,
        'version': '003'
    })
    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '003',
        'lang_code': 'en',
        'model_name': 'gpt-5-mini',
        'content_blocks': 'step_get_en_content_blocks'
    })
    map_reduce_dict['step_generate_facts_fr'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '002',
        'lang_code': 'fr', 
        'model_name': 'gpt-5-mini',
        'content_blocks': 'step_get_fr_content_blocks'
    })
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'fr_facts': 'step_generate_facts_fr'
    })
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs'
    })
    map_reduce_dict['step_find_retrieval_candidates'] = SingletonStep(step_retrieve_potential_matches, {
        'version': '010',
        'en_facts': 'step_generate_facts',
        'fr_facts': 'step_generate_facts_fr',
        'alignment_df': 'step_union_fact_paragraphs'
    })
    # TODO: need to update the prompt here.
    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': '006',
        'model_name': 'gpt-5-mini',
        'info_gap_retrieval_dfs': 'step_find_retrieval_candidates'
    })
    map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(step_collapse_gpt_labels, {
        'version': '002',
        'model_intersection_names': ('gpt-5-mini',), 
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    })
    return map_reduce_dict