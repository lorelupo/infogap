

def get_en_tgt_info_diff_map_dict(en_bio_id=None, tgt_bio_id=None, person_name=None, tgt_person_name=None, tgt_lang=None):
    map_reduce_dict = OrderedDict()

    ## English
    en_bio_id_dict = {'en_bio_id': en_bio_id} if en_bio_id else {}
    tgt_bio_id_dict = {'tgt_bio_id': tgt_bio_id} if tgt_bio_id else {}
    person_name_dict = {'person_name': person_name, 'tgt_person_name': tgt_person_name} if person_name else {}
    print(tgt_bio_id_dict)
    map_reduce_dict['step_get_en_content_blocks'] = SingletonStep(step_retrieve_prescraped_content_blocks, { # in info diff steps
        'version': '003', 
        'lang': 'en',
        **en_bio_id_dict
    },)
    map_reduce_dict['step_get_tgt_content_blocks'] = SingletonStep(step_retrieve_prescraped_content_blocks, { # in info diff steps
        'version': '003', 
        'lang': tgt_lang,
        **tgt_bio_id_dict
    })
    ## Repeat, but for chinese
    map_reduce_dict['step_generate_facts'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '003',
        'lang_code': 'en',
        'content_blocks': 'step_get_en_content_blocks', 
        **person_name_dict
    })
    map_reduce_dict['step_generate_facts_tgt'] = SingletonStep(step_generate_facts, { # in info diff steps
        'version': '002',
        'lang_code': tgt_lang, 
        'content_blocks': 'step_get_tgt_content_blocks', 
        **person_name_dict
    })
    # map_reduce_dict['step_infer_pronoun'] = SingletonStep(step_infer_pronoun, {
    #     'version': '001',
    #     'en_content_blocks': 'step_get_en_content_blocks' 
    # })
    #### 
    map_reduce_dict['step_align_fact_paragraphs'] = SingletonStep(step_obtain_en_tgt_paragraphs_associations, {
        'version': '003',
        'en_facts': 'step_generate_facts',
        'tgt_facts': 'step_generate_facts_tgt'
    })
    map_reduce_dict['step_union_fact_paragraphs'] = SingletonStep(step_union_alignments, {
        'version': '002',
        'unpruned_alignment_strns': 'step_align_fact_paragraphs',
        'lang_code': tgt_lang,
    })

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
    ### 

    map_reduce_dict['step_reasoning_intersection_label'] = SingletonStep(step_compute_info_gap_reasoning, {
        'version': '006',
        'model_name': 'gpt-5-mini',
        'lang_code': tgt_lang,
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
        'tgt_content_blocks': 'step_get_tgt_content_blocks',
        'pronoun': None
    })
    return map_reduce_dict





