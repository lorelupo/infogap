import openai
import loguru
import ipdb
from typing import List, Tuple, Dict
# import dataclasses    
from dataclasses import dataclass
from packages.constants import LANG_MAPPINGS, ASK_GPT_FACT_EXTRACTION_PROMPTS, ASK_GPT_FACT_INTERSECTION_PROMPTS

logger = loguru.logger

@dataclass
class FactParagraph:
    """
    A list of facts that are contained in a Wikipedia paragraph.
    """ 
    facts: List[str]

    def __len__(self):
        return len(self.facts)
    
    # make it iterable
    def __iter__(self):
        return iter(self.facts)

def load_tsvetshop_client():
    """Deprecated: prefer load_other_client() which supports multiple providers."""
    try:
        from packages.steps.info_diff_steps import load_other_client
        return load_other_client()
    except Exception:
        # Fallback to default OpenAI client using environment resolution in SDK
        return openai.OpenAI()


def _create_chat_completion(client, *, model: str, messages: list, temperature: float | None = None, max_tokens: int | None = None, **extra):
    """
    Wrapper for client.chat.completions.create that gracefully handles providers
    that don't support 'max_tokens' and require 'max_completion_tokens' instead,
    and providers that reject temperature=0.
    """
    # Model-specific parameter handling
    model_lower = (model or "").lower()
    is_gpt5_family = model_lower in ("gpt-5", "gpt-5-mini")

    # Build base params
    params = {"model": model, "messages": messages, **extra}

    # Tokens: use provider-preferred field
    if max_tokens is not None:
        if is_gpt5_family:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    # Temperature: omit for gpt-5 family; include otherwise if explicitly provided
    if not is_gpt5_family and temperature is not None:
        params["temperature"] = temperature
    return client.chat.completions.create(**params)

def construct_fact_decomp_prompt(src_lang, paragraph: str):
    if src_lang == 'en':
        input_prompt = f"Please breakdown the following paragraph into a list of independent facts. All of the facts should be placed in a stringified python list.\n {paragraph}"
    elif src_lang == 'fr':
        input_prompt = f"Veuillez décomposer le paragraphe suivant en une liste de faits indépendants. Tous les faits doivent être placés dans une liste python sous forme de chaîne de caractères.\n {paragraph}"
    elif src_lang == 'ru':
        input_prompt = f"Пожалуйста, разбейте следующий абзац на список независимых фактов. Все факты должны быть помещены в строковый список Python (e.g., ['Тим вырос в городе Мальорке, штат Алабама.','Его отец был работником верфи.', 'Мать Тима была домохозяйкой.','Кук получил степень бакалавра в области промышленного производства в университете Обёрна в 1982 году.','Кук получил диплом МВА в школе Фукуа университета Дьюка в 1988 году.']).\n {paragraph}"
    else: 
        raise ValueError(f"Invalid language: {src_lang}")
    return input_prompt

# def ask_gpt_for_facts(client, 
#                       model_name: str,
#                       paragraph: str, lang_code: str):
#     # message=[{"role": "user", "content": f"Please breakdown the following sentence into independent facts. Return: {sentence}"}]
#     if lang_code == 'en':
#         message=[{"role": "user", "content": f"Please breakdown the following paragraph into a list of independent facts. All of the facts should be placed in a stringified python list.\n {paragraph}"}]
#     elif lang_code == 'fr':
#         message = [{"role": "user", "content": f"Veuillez décomposer le paragraphe suivant en une liste de faits indépendants. Tous les faits doivent être placés dans une liste python sous forme de chaîne de caractères.\n {paragraph}"}]
#     elif lang_code == 'ru':
#         message = [{"role": "user", "content": f"Пожалуйста, разбейте следующий абзац на список независимых фактов. Все факты должны быть помещены в строковый список Python (e.g., ['Тим вырос в городе Мальорке, штат Алабама.','Его отец был работником верфи.', 'Мать Тима была домохозяйкой.','Кук получил степень бакалавра в области промышленного производства в университете Обёрна в 1982 году.','Кук получил диплом МВА в школе Фукуа университета Дьюка в 1988 году.']).\n {paragraph}"}]
#     elif lang_code == 'zh':
#         message = [{"role": "user", "content": f"请将以下段落分解为一系列独立事实。所有事实都应放在字符串化的 Python 列表中。\n {paragraph}"}]
#     elif lang_code == 'ja':
#         message = [{"role": "user", "content": f"以下の段落を独立した事実の列に分解し、すべての事実は文字列化された Python リストに配置する。 \n {paragraph}"}]
#     elif lang_code == 'ko':
#         message = [{"role": "user", "content": f"다음 단락을 독립적인 사실 목록으로 분류해 주십시오. 모든 사실은 문자열화된 Python 목록에 배치되어야 합니다. \n {paragraph}"}]
#     elif lang_code == 'he':
#         message = [{"role": "user", "content": f"אנא חלק את הפסקה הבאה לרשימה של עובדות עצמאיות. יש למקם את כל העובדות ברשימת פיתון מחורזת. \n {paragraph}"}]
#     response = client.chat.completions.create(
#         model=model_name,
#         temperature=0,
#         max_tokens = len(paragraph) + 1000,
#         messages = message)
#     # response = client.chat.completions.create(
#     #     model="gpt-4",
#     #     max_tokens=len(paragraph) + 100,
#     #     temperature=0,
#     # messages = message)
#     response_content = response.choices[0].message.content
#     logger.info(f"RRRResponse_content: {response_content}")
#     response_total_tokens = response.usage.total_tokens
#     return response_content, response_total_tokens


def ask_gpt_for_facts(client, model_name: str, paragraph: str, lang_code: str):
    """
    Requests GPT to break down a paragraph into a list of independent facts.

    Args:
        client: The GPT client for making API requests.
        model_name (str): The GPT model name.
        paragraph (str): The paragraph to be processed.
        lang_code (str): The language code for the paragraph.

    Returns:
        Tuple[str, int]: A tuple containing the response content and the total token usage.
    """
    # Validate language code and fetch the corresponding prompt
    if lang_code not in ASK_GPT_FACT_EXTRACTION_PROMPTS:
        raise ValueError(f"Unsupported language code: {lang_code}")

    input_prompt = ASK_GPT_FACT_EXTRACTION_PROMPTS[lang_code].format(paragraph=paragraph)
    message = [{"role": "user", "content": input_prompt}]

    response = _create_chat_completion(
        client,
        model=model_name,
        temperature=0,
        max_tokens=len(paragraph) + 1000,
        messages=message,
    )

    response_content = response.choices[0].message.content
    # logger.info(f"GPT Response: {response_content}")

    return response_content, response.usage.total_tokens

def construct_fact_intersection_prompt(src_lang: str, tgt_lang_code: str, src_fact_context: List[str], 
                                       tgt_fact_context: List[str], person_name: str):
    en_tgt_lang_code_to_lang = {'en': 'English', 'fr': 'French', 'ru': 'Russian'}
    ru_lang_code_to_lang = {'en': 'английский', 'fr': 'французский', 'ru': 'русский'}
    if src_lang == 'en':
        input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the list inferrerable from the following {en_tgt_lang_code_to_lang[tgt_lang_code]} facts?\n {tgt_fact_context}. Return either 'yes' or 'no'."
    elif src_lang == 'fr':
        input_prompt = f"Considérez ces faits français sur {person_name}:\n {src_fact_context}. Est le dernier fait de la liste inférable de l'une des listes de faits suivantes?\n {tgt_fact_context}. Retournez 'oui' ou 'non'."
    elif src_lang == 'ru':
        input_prompt = f"Рассмотрим эти факты на русском языке о {person_name}:\n {src_fact_context}. Можно ли вывести последний факт из одного из следующих списков фактов?\n {tgt_fact_context} Возвращает список, содержащий ['да' или 'нет'] — один ответ для каждого списка фактов {ru_lang_code_to_lang[tgt_lang_code]}. Все ответы «да/нет» должны быть помещены в список строк Python. (например, ['да', 'нет', 'да'])"
    # TODO: add the Russian version of the prompt.
    return input_prompt

# def ask_gpt_about_fact_intersection(client, model_name, cache,  
#                                     src_lang_code, 
#                                     tgt_lang_code,
#                                     src_fact_context: List[str], tgt_fact_context: List[List[str]], 
#                                     person_name: str, 
#                                     tgt_person_name: str):
#     en_lang_code_to_lang = {'en': 'English', 'fr': 'French', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'he': 'Hebrew'}
#     fr_lang_code_to_lang = {'en': 'anglais', 'fr': 'français', 'ru': 'russe', 'zh': 'chinois', 'ja': 'japonais', 'ko': 'coréen', 'he': 'hébraïque'}
#     ru_lang_code_to_lang = {'en': 'английский', 'fr': 'французский', 'ru': 'русский', 'zh': 'русский', 'ja': 'японский', 'ko': 'корейский', 'he': 'иврит'}
#     zh_lang_code_to_lang = {'en': '英语', 'fr': '法语', 'ru': '俄语', 'zh': '中文', 'ja': '日语', 'ko': '韓语', 'he': '希伯来语'}
#     ja_lang_code_to_lang = {'en': '英語', 'fr': '法語', 'ru': 'ルス語', 'zh': '中国語', 'ja': '日本語', 'ko': '韓国語', 'he': 'ヘブライ語'}
#     ko_lang_code_to_lang = {'en': '영어', 'fr': '프ラン쪝스', 'ru': '루시아', 'zh': '최종 트어', 'ja': '일반어', 'ko': '코레이션', 'he': '흩어'}
#     he_lang_code_to_lang = {'en': 'אנגלית', 'fr': 'צרפתית', 'ru': 'רוסית', 'zh': 'סינית', 'ja': 'יפנית', 'ko': 'קוריאנית', 'he': 'עברית'}

#     if src_lang_code == 'en':
#         # input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the list inferrerable from the following French facts?\n {tgt_fact_context}. Return either 'yes' or 'no'."
#         input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the former list inferrable from any of these lists of facts?\n {tgt_fact_context} Return a list containing ['yes' or 'no'] -- one response for each list of {en_lang_code_to_lang[tgt_lang_code]} facts. All of the yes/no responses should be placed in a stringified python list. (e.g., ['yes', 'no', 'yes'])"
#     elif src_lang_code == 'fr':
#         input_prompt = f"Considérez ces faits français sur {person_name}:\n {src_fact_context}. Est le dernier fait de la liste inférable de l'une des listes de faits suivantes?\n {tgt_fact_context} Retournez une liste contenant ['yes' ou 'no'] -- une réponse pour chaque liste de faits {fr_lang_code_to_lang[tgt_lang_code]}. Toutes les réponses yes/no doivent être placées dans une liste python. (e.g., ['yes', 'no', 'yes'])"
#     elif src_lang_code == 'ru':
#         input_prompt = f"Рассмотрим эти факты на русском языке о {tgt_person_name}:\n {src_fact_context}. Можно ли вывести последний факт из одного из следующих списков фактов?\n {tgt_fact_context} Возвращает список, содержащий ['да' или 'нет'] — один ответ для каждого списка фактов {ru_lang_code_to_lang[tgt_lang_code]}. Все ответы «да/нет» должны быть помещены в список строк Python. (например, ['да', 'нет', 'да'])"
#     elif src_lang_code == 'zh':
#         input_prompt = f"考虑以下关于 {tgt_person_name} 的中文事实：\n {src_fact_context}。前一个列表中的最后一个事实是否可以从以下这些事实列表中推断出来？\n {tgt_fact_context} 返回一个包含 ['yes' 或 'no'] 的列表 -- 每个 {zh_lang_code_to_lang[tgt_lang_code]} 事实列表都有一个 'yes' 或 'no' 响应。所有 'yes' 或 'no' 响应都应该放置在一个字符串化的 Python 列表中 (比如, ['yes', 'no', 'yes'])"
#     elif src_lang_code == 'ja':
#         input_prompt = f"以下の {person_name} に関する英語の事実を考慮してください:\n {src_fact_context}. このリストの最後の事実は、次のいずれかの事実リストから推論できますか?\n {tgt_fact_context} 各 {ja_lang_code_to_lang[tgt_lang_code]} の事実リストに対して [‘yes’ または ‘no’] の応答リストを返してください。すべての yes/no の応答は、文字列化された Python リストの形式で返す必要があります。（例: [‘yes’, ‘no’, ‘yes’]）"
#     elif src_lang_code == 'ko':
#         input_prompt = f"다음 {person_name}에 대한 영어 사실들을 고려하세요:\n {src_fact_context}. 이전 목록의 마지막 사실이 다음 사실 목록 중 어느 하나로부터 추론될 수 있습니까?\n {tgt_fact_context} 각 {ko_lang_code_to_lang[tgt_lang_code]} 사실 목록에 대해 [‘yes’ 또는 ‘no’]로 구성된 응답 리스트를 반환하세요. 모든 yes/no 응답은 문자열 형태의 Python 리스트로 반환해야 합니다. (예: [‘yes’, ‘no’, ‘yes’])"
#     elif src_lang_code == 'he':
#         input_prompt = f"שקול את העובדות האלה באנגלית על {person_name}:\n {src_fact_context}. האם ניתן להסיק את העובדה האחרונה ברשימה הקודמת מכל אחת מרשימות העובדות הללו?\n {tgt_fact_context} החזר רשימה המכילה ['כן' אוzלא'] -- תגובה אחת לכל רשימה של {en_lang_code_to_lang[tgt_lang_code]} עובדות. יש למקם את כל התשובות כן/לא במחרוזת המייצגת רשימה בפייתון. (לדוגמה, ['כן', 'לא', 'כן'])"
#     else:
#         raise ValueError(f"Invalid language code: {src_lang_code}")
#     if input_prompt not in cache:
#         message=[{"role": "user", "content": input_prompt}]
#         # TODO: implement this.
#         def start_debug():
#             logger.error("Model name is not in the list of valid models.")
#             ipdb.set_trace()
#         assert model_name in ['gpt4v', 'gpt-4', 'gpt-5-mini', 'gpt-3.5-turbo-0125'], start_debug()
#         response = client.chat.completions.create(
#             model=model_name,
#             max_tokens=len(src_fact_context) + len(tgt_fact_context) + 2000,
#             temperature=0,
#         messages = message)
#         # .choices[0].message.content
#         response_content = response.choices[0].message.content
#         if src_lang_code == 'ru':
#             response_content = response_content.replace('да', 'yes').replace('нет', 'no')
#         if src_lang_code == 'he':
#             response_content = response_content.replace('כן', 'yes').replace('לא', 'no')
#         response_total_tokens = response.usage.total_tokens
#         cache[input_prompt] = response_content
#         return input_prompt, response_content, response_total_tokens
#     else:
#         return input_prompt, cache[input_prompt], 0 


def format_fact_context(fact_context: List[str], lang_code: str) -> str:
    formatted_facts = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(fact_context))
    return formatted_facts

def ask_gpt_about_fact_intersection(client, model_name, cache,  
                                    src_lang_code: str, tgt_lang_code: str,
                                    src_fact_context: List[str], tgt_fact_context: List[List[str]], 
                                    person_name: str, tgt_person_name: str):

    # Validate and fetch language names
    src_language_map = LANG_MAPPINGS.get(src_lang_code, {})
    tgt_language = src_language_map.get(tgt_lang_code, tgt_lang_code)

    # Validate source language
    if src_lang_code not in ASK_GPT_FACT_INTERSECTION_PROMPTS:
        raise ValueError(f"Invalid source language code: {src_lang_code}")

    formated_source_fact_context = format_fact_context(src_fact_context, src_lang_code)
    # ipdb.set_trace()
    if len(tgt_fact_context) == 1:
        formated_tgt_fact_context = format_fact_context(tgt_fact_context[0], tgt_lang_code)
    else:
        logger.error(f"Invalid tgt_fact_context: {tgt_fact_context}, length: {len(tgt_fact_context)}")
    if src_lang_code == 'en':
        person_name = person_name
    else:
        person_name = tgt_person_name
    input_prompt = ASK_GPT_FACT_INTERSECTION_PROMPTS[src_lang_code].format(
        person_name=person_name,
        tgt_person_name=tgt_person_name,
        src_fact_context=formated_source_fact_context,
        tgt_fact_context=formated_tgt_fact_context,
        tgt_language=tgt_language
    )
    # print(input_prompt)
    # Check if the input prompt is in the cache
    if input_prompt not in cache:
        message = [{"role": "user", "content": input_prompt}]
        
        assert model_name in ['gpt4v', 'gpt-4', 'gpt-4o', 'gpt-5', 'gpt-5-mini', 'gpt-3.5-turbo-0125'], "Invalid model name"

        response = _create_chat_completion(
            client,
            model=model_name,
            max_tokens=len(src_fact_context) + len(tgt_fact_context) + 2000,
            temperature=0,
            messages=message,
        )

        response_content = response.choices[0].message.content
        response_total_tokens = response.usage.total_tokens

        # Normalize Yes/No responses for Russian and Hebrew
        response_content = response_content.replace("да", "yes").replace("нет", "no") if src_lang_code == "ru" else response_content
        response_content = response_content.replace("כן", "yes").replace("לא", "no") if src_lang_code == "he" else response_content

        cache[input_prompt] = response_content
        return input_prompt, response_content, response_total_tokens
    
    return input_prompt, cache[input_prompt], 0

def ask_gpt_if_premise_entails_hypothesis(premise: str, hypothesis: str):
    message=[{"role": "user", "content": f"Please determine if the following premise entails the following hypothesis. Return either 'entails', 'contradicts', or 'neither'.\n Premise: {premise}\n Hypothesis: {hypothesis}"}]
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        max_tokens=len(premise) + len(hypothesis) + 1000,
        temperature=0.8,
    messages = message)
    return response

def ask_gpt_about_caa_classification(client, content: str, person_name: str ):
    # "{text}. Is the sentiment towards {word} in the given sentence positive, neutral or negative?"
    # "{text}. Is the implied sentiment towards {word} in the given text positive, neutral or negative?"
    # "{text}. Is the tone towards {word} in the given text positive, neutral or negative?"
    ## NOTE: display vs. imply
    prompt_implied_sent = f"Consider the following text: \n {content}\n. Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    message = [{"role": "user", "content": prompt_implied_sent}]
    response = _create_chat_completion(
        client,
        model="gpt-4",
        temperature=None,
        messages=message,
    ).choices[0].message.content
    if response not in ['pos', 'neutral', 'neg', 'none']:
        logger.warning(f"Invalid response from GPT-3: [[{response}]] for prompt:\n\n {prompt_implied_sent}")
    return response

def ask_gpt_about_caa_classification_coreference_resolution(client, content: str, person_name: str, pronoun: str):
    coref_string = f"{person_name} identifies as {pronoun}. In case of ambiguity, the pronoun {pronoun} refers to {person_name}."
    prompt_implied_sent = f"Consider the following text: \n {content}\n. {coref_string} Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    message = [{"role": "user", "content": prompt_implied_sent}]

    response = _create_chat_completion(
        client,
        model="gpt-4",
        temperature=None,
        messages=message,
    ).choices[0].message.content
    if response not in ['pos', 'neutral', 'neg', 'none']:
        logger.warning(f"Invalid response from GPT-3: [[{response}]] for prompt:\n\n {prompt_implied_sent}")
    return response

def prep_caa_prompt(language, content: List[str], pronoun, person_name) -> str:
    if language == 'en':
        coref_string = f"{person_name} identifies as {pronoun}. In case of ambiguity, the pronoun {pronoun} refers to {person_name}."
        prompt_implied_sent = f"Consider the last sentence in the following text: \n {content}\n. {coref_string} Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    elif language == 'fr':
        coref_string = f"{person_name} s'identifie comme {pronoun}. En cas d'ambiguïté, le pronom {pronoun} se réfère à {person_name}."
        prompt_implied_sent = f"Considérez la dernière phrase du texte suivant: \n {content}\n. {coref_string} Est-ce que ce contenu implique un sentiment positif, neutre ou négatif envers {person_name}? (pos/neutre/neg/none)"
    else:
        raise ValueError(f"Invalid language: {language}")
    return prompt_implied_sent

def prompt_gpt_4(client, valid_labels: List[str], prompt) -> Tuple[str, int]:
    message = [{"role": "user", "content": prompt}]
    response = _create_chat_completion(
        client,
        model="gpt-4",
        temperature=None,
        max_tokens=len(prompt) + 1000,
        messages=message,
    )
    response_content = response.choices[0].message.content
    if response_content not in valid_labels:
        logger.warning(f"Invalid response from GPT-4: [[{response_content}]] for prompt:\n\n {prompt}")
    response_total_tokens = response.usage.total_tokens
    return response_content, response_total_tokens

