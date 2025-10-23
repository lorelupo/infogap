# Utilizing InfoGap Pipeline and Developing New Output Processing for WikiGap Project (CSCW 2026)

This repository contains the code for running the **InfoGap** pipeline for the **WikiGap CSCW 2026** paper.

> **Link to the WikiGap repo:** <https://github.com/aw814/WikiGap>  
> This branch is solely for generating the datasets used in the WikiGap web extension.  
> The **final output files** should be uploaded to the `json/` folder in the WikiGap repo.

---

## 📬 Contact

`fsamir@mail.ubc.ca` · `zining.wang@ubc.ca`

---

## 🏃‍♂️ Running the Pipeline

Follow the steps below to execute the InfoGap pipeline.

### 1. Set up an Environment file

Create a `.env` file in the project root directory and add **two** environment variables:

| Variable      | Description                                               |
|---------------|-----------------------------------------------------------|
| `SCRATCH_DIR` | Directory where the pipeline will store its artifacts     |
| `THE_KEY`     | Your OpenAI API key                                       |

### 2. Install requirements

```bash
source .venv/bin/activate    # (Linux/macOS)
# OR
.venv\Scripts\activate       # (Windows)

pip install -r requirements.txt
```

### 3. Scrape the articles you want

1.  Modify **`wikigap_topics_scrape.py`** to include the topics you want to scrape.  
2.  Run the scraper:
    ```bash
    python main_scrape_bios.py scrape-bios
    ```
3.  When prompted, enter the **target language code** (e.g., `fr`, `ru`, `zh`).  
    The scraped results are saved to the path specified by `SCRATCH_DIR`.

### 4. Run the InfoGap pipeline

```bash
python main_complete_analysis.py run-multiple-topics
```

* The script reads pairs of scraped article titles from `scraped_titles_{lang}.py`.  
* It then analyzes the scraped articles and generates the output.
* **Very important:** please set the `TGT_LANG` variable in `constants.py` to the target language you want to analyze. `TGT_LANG` is used to retrieve the target language content blocks. 

---

## 🔬 Pipeline Overview

The core logic lives in `packages/steps/map_dicts.py`, function **`get_en_tgt_info_diff_map_dict`**, which builds a sequence of **SingletonSteps** to:

1. Retrieve pre‑scraped content blocks  
2. Generate facts from each article  
3. Align and union fact–paragraph associations  
4. Identify cross‑lingual fact matches  
5. Apply GPT‑based reasoning to detect information gaps  

### Example Steps

#### Retrieve Target‑Language Content Blocks

```python
map_reduce_dict['step_get_tgt_content_blocks'] = SingletonStep(
    step_retrieve_prescraped_tgt_content_blocks,
    {
        'version': '003',
        **tgt_bio_id_dict,
        **tgt_lang_dict
    }
)
```

#### Generate Facts with LLM for Target Language

```python
map_reduce_dict['step_generate_facts_tgt'] = SingletonStep(
    step_generate_facts,
    {
        'version': '002',
        'lang_code': tgt_lang,
        'content_blocks': 'step_get_tgt_content_blocks',
        **person_name_dict
    }
)
```

#### Collapse GPT Labels

```python
map_reduce_dict['step_collapse_gpt_labels'] = SingletonStep(
    step_collapse_gpt_labels,
    {
        'version': '002',
        'model_intersection_names': ('gpt-4o',),
        'gpt_info_gap_dfs': 'step_reasoning_intersection_label'
    }
)
```

The final step produces a **binary label** for each fact indicating whether it exists in **both** language editions (`yes`) or **only in one** (`no`).

---

## ⚙️ Pipeline Execution

The entire pipeline is executed via `run_complete_gpt_pipeline`, which internally calls the map‑reduce steps:

```python
metadata = conduct(
    os.path.join(SCRATCH_DIR, f"full_cache_gpt_en_{tgt_lang}"),
    full_map_dict,
    f"en_{tgt_lang}_gpt_logs"
)
```

You can then load the resulting artifacts with:

```python
info_gap_dfs = load_mr_artifact(metadata[0])
```

`info_gap_dfs` is a tuple of three **Polars DataFrames**:

| Index | Direction                         | Note            |
|-------|-----------------------------------|-----------------|
| 0     | English → Target Language         |                 |
| 1     | Target Language → English         |                 |
| 2     | Legacy placeholder                | Ignored in WikiGap |

Each DataFrame includes a **`gpt-4o_intersection_label`** column:

* `yes` – fact is found in **both** language editions  
* `no`  – fact is found **only** in the source language  

---

## 📁 Output Location

Step 3 of `run_complete_gpt_pipeline` saves `info_gap_dfs[0]` and `info_gap_dfs[1]` to `ethnic_annotation_save/wikigap_data/` as JSON files named:

```
annotation_<generation_date>_<article_topic>_<lang>.json
```

Example: `annotation_2024-03-24_Peking_Duck_fr.json`

> **Debug tip:** Each step’s intermediate output is cached under `${SCRATCH_DIR}/full_cache/` via **flowmason**.

---

## VII. Process Annotations and Format InfoGap Output for WikiGap  
*(New process developed for the WikiGap project)*

Because WikiGap relies entirely on automatic knowledge alignment using LLMs, we add a **post‑processing** step:

```bash
python process_annotations.py
```

### What `process_annotations.py` does

1. **Load** the annotation JSONs produced by InfoGap.  
2. **Retrieve** paragraph blocks for English and target‑language articles.  
3. **Match** each fact to its paragraph and section headers.  
4. **Filter** for language‑specific facts (`intersection_label == "no"`).  
5. **Translate** headers (Google) and facts (GPT‑4o) into English.  
6. Optionally **sample** facts per header.  
7. **Save** a nested `{topic}.json` file for the WikiGap extension.

#### Input / Output paths

| Stage | Path pattern |
|-------|--------------|
| Input annotation JSONs | `scratch/annotation_save/wikimt_data/annotation_{date}_{en_title}_{lang}.json` |
| Paragraph block pickles | `scratch/article_contents/{bio_id}_{lang}.pkl` |
| Final WikiGap JSONs | `scratch/annotation_save/wikimt_data/json/{topic}.json` |

#### Running the script

1. Ensure `packages.scraped_titles_{lang}.py` exists and contains `en_tgt_title_pairs`.  
2. Ensure `wikigap_topics_scrape.py` has an up‑to‑date `selected_topics` list.  
3. Set environment variables for Azure OpenAI:

```bash
export THE_KEY=your-api-key
```

Then run:

```bash
python process_annotations.py
```

##### Optional sampling

To sample a subset of facts per header (e.g., 15):

```python
df_tgt_sampled = weighted_sampling_by_header(
    df_filtered,
    header_column="header_1",
    sample_size=15
)
```

---

## ➡️ Using the Output with the WikiGap Extension

Copy the generated `{topic}.json` files into the WikiGap repo’s **`json/`** directory:

```
WikiGap/json/{topic}.json
```

---

## 📜 Citation

```bibtex
@inproceedings{samir-2024-information,
  title     = {Locating Information Gaps and Narrative Inconsistencies Across Languages: A Case Study of LGBT People Portrayals on Wikipedia},
  author    = {Samir, Farhan and Park, Chan Young and Field, Anjalie and Shwartz, Vered and Tsvetkov, Yulia},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024},
  address   = {Miami},
  publisher = {Association for Computational Linguistics}
}
```
