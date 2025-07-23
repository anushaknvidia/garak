# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Translator that translates a prompt."""


from typing import List, Callable
import re
import unicodedata
import string
import logging
from garak.resources.api import nltk
from langdetect import detect, DetectorFactory, LangDetectException

_intialized_words = False


def _initialize_words():
    global _intialized_words
    if not _intialized_words:
        # Ensure the NLTK words corpus is downloaded
        try:
            nltk.data.find("corpora/words")
        except LookupError as e:
            nltk.download("words", quiet=True)
        _intialized_words = True


def remove_english_punctuation(text: str) -> str:
    punctuation_without_apostrophe = string.punctuation.replace("'", "")
    return " ".join(
        re.sub(":|,", "", char)
        for char in text
        if char not in punctuation_without_apostrophe
    )


def is_english(text):
    """Determines if the given text is predominantly English based on word matching.

    Args:
        text (str): The text to evaluate.

    Returns:
        bool: True if more than 50% of the words are English, False otherwise.
    """
    # Load English words from NLTK
    _initialize_words()
    from nltk.corpus import words

    special_terms = {"ascii85", "encoded", "decoded", "acsii", "plaintext"}
    english_words = set(words.words()).union(special_terms)

    text = text.lower()
    word_list = text.split()
    if len(word_list) == 0:
        return False

    if len(word_list) >= 1:
        word_list = remove_english_punctuation(word_list)
    else:
        word_list = word_list[0]

    if word_list:
        word_list = word_list.split()
        cleaned_words = " ".join(char for char in word_list if char.isalpha())
        # Filter out empty strings
        cleaned_words = cleaned_words.split()
        cleaned_words = [word for word in cleaned_words if word]

        if not cleaned_words:
            return False

        english_word_count = sum(1 for word in cleaned_words if word in english_words)
        return (english_word_count / len(cleaned_words)) > 0.5
    return False


def split_input_text(input_text: str) -> list:
    """Split input text based on the presence of ': '."""
    if (
        input_text is not None
        and ": " in input_text
        and "http://" not in input_text
        and "https://" not in input_text
    ):
        split_text = input_text.splitlines()
        split_text = [line.split(":") for line in split_text]
        split_text = [item for sublist in split_text for item in sublist]
    else:
        split_text = input_text.splitlines()
    return split_text


def contains_invisible_unicode(text: str) -> bool:
    """Determine whether the text contains invisible Unicode characters."""
    if not text:
        return False
    for char in text:
        if unicodedata.category(char) not in {"Cc", "Cf", "Cn", "Zl", "Zp", "Zs"}:
            return False
    return True


def is_meaning_string(text: str) -> bool:
    """Check if the input text is a meaningless sequence or invalid for translation."""
    DetectorFactory.seed = 0

    # Detect Language: Skip if no valid language is detected
    try:
        lang = detect(text)
    except LangDetectException:
        logging.debug("langdetect failed to detect a valid language.")
        return False

    if lang == "en":
        return False

    # Length and pattern checks: Skip if it's too short or repetitive
    if len(text) < 3 or re.match(r"(.)\1{3,}", text):  # e.g., "aaaa" or "123123"
        return False

    return True


def preserve_special_tags(text: str) -> tuple[str, dict]:
    """
    Preserve special tags and bracketed content during translation.
    
    Args:
        text (str): The text to process
        
    Returns:
        tuple: (processed_text, tag_mapping) where processed_text has tags replaced
               with placeholders and tag_mapping contains the original tags
    """
    # Define patterns for special tags that should be preserved
    # This includes [/INST], [INST], [SYS], </SYS>, and other common instruction tags
    special_tag_patterns = [
        r'\[/INST\]',  # End instruction tag
        r'\[INST\]',   # Start instruction tag
        r'\[SYS\]',    # System tag
        r'\[/SYS\]',   # End system tag
        r'<<SYS>>',    # Alternative system tag format
        r'<</SYS>>',   # Alternative end system tag format
        r'\[/INST\]',  # Alternative end instruction tag
        r'\[INST\]',   # Alternative start instruction tag
        r'\[/SYS\]',   # Alternative end system tag
        r'\[SYS\]',    # Alternative start system tag
        r'\[/INST\]',  # Another variant
        r'\[INST\]',   # Another variant
        r'\[/SYS\]',   # Another variant
        r'\[SYS\]',    # Another variant
    ]
    
    # Also preserve any content in square brackets that looks like instructions
    bracket_pattern = r'\[[^\]]*\]'
    
    # Preserve JSON-like structures that might contain important metadata
    json_pattern = r'\{[^}]*"improvement"[^}]*"prompt"[^}]*\}'
    
    # Preserve backtick-wrapped content (code blocks, etc.)
    backtick_pattern = r'`[^`]*`'
    
    # Preserve asterisk-wrapped content (bold text, etc.)
    asterisk_pattern = r'\*\*[^*]*\*\*'
    
    tag_mapping = {}
    processed_text = text
    placeholder_counter = 0
    
    # First, preserve special instruction tags
    for pattern in special_tag_patterns:
        matches = re.finditer(pattern, processed_text, re.IGNORECASE)
        for match in matches:
            placeholder = f"__SPECIAL_TAG_{placeholder_counter}__"
            tag_mapping[placeholder] = match.group(0)
            processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
            placeholder_counter += 1
    
    # Preserve JSON-like structures
    json_matches = re.finditer(json_pattern, processed_text)
    for match in json_matches:
        placeholder = f"__JSON_TAG_{placeholder_counter}__"
        tag_mapping[placeholder] = match.group(0)
        processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
        placeholder_counter += 1
    
    # Preserve backtick-wrapped content
    backtick_matches = re.finditer(backtick_pattern, processed_text)
    for match in backtick_matches:
        placeholder = f"__BACKTICK_TAG_{placeholder_counter}__"
        tag_mapping[placeholder] = match.group(0)
        processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
        placeholder_counter += 1
    
    # Preserve asterisk-wrapped content
    asterisk_matches = re.finditer(asterisk_pattern, processed_text)
    for match in asterisk_matches:
        placeholder = f"__ASTERISK_TAG_{placeholder_counter}__"
        tag_mapping[placeholder] = match.group(0)
        processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
        placeholder_counter += 1
    
    # Then preserve any remaining bracketed content that might be important
    bracket_matches = re.finditer(bracket_pattern, processed_text)
    for match in bracket_matches:
        # Skip if this was already processed as a special tag
        if any(placeholder in processed_text[match.start():match.end()] for placeholder in tag_mapping.keys()):
            continue
        
        # Check if the bracketed content looks like it should be preserved
        content = match.group(0)
        # Preserve if it contains common instruction-related words or looks like a tag
        if any(keyword in content.lower() for keyword in ['inst', 'sys', 'user', 'assistant', 'human', 'ai', 'model', 'new prompt', 'task here', 'banned word']):
            placeholder = f"__BRACKET_TAG_{placeholder_counter}__"
            tag_mapping[placeholder] = content
            processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
            placeholder_counter += 1
    
    return processed_text, tag_mapping


def restore_special_tags(text: str, tag_mapping: dict) -> str:
    """
    Restore special tags that were preserved during translation.
    
    Args:
        text (str): The translated text with placeholders
        tag_mapping (dict): Mapping of placeholders to original tags
        
    Returns:
        str: Text with placeholders replaced by original tags
    """
    restored_text = text
    for placeholder, original_tag in tag_mapping.items():
        restored_text = restored_text.replace(placeholder, original_tag)
    return restored_text


# To be `Configurable` the root object must meet the standard type search criteria
# { langproviders:
#     "local": { # model_type
#       "language": "<from>-<to>"
#       "name": "model/name" # model_name
#       "hf_args": {} # or any other translator specific values for the model_type
#     }
# }
from garak.configurable import Configurable


class LangProvider(Configurable):
    """Base class for objects that provision language"""

    DEFAULT_PARAMS = {
        "preserve_special_tags": True,  # Whether to preserve special tags during translation
    }

    def __init__(self, config_root: dict = {}) -> None:

        self._load_config(config_root=config_root)

        self.source_lang, self.target_lang = self.language.split(",")

        self._validate_env_var()

        self._load_langprovider()

    def _load_langprovider(self):
        raise NotImplementedError

    def _translate(self, text: str) -> str:
        raise NotImplementedError

    def _get_response(self, input_text: str):
        # Preserve special tags before translation if enabled
        if getattr(self, 'preserve_special_tags', True):
            processed_text, tag_mapping = preserve_special_tags(input_text)
        else:
            processed_text = input_text
            tag_mapping = {}
        
        translated_lines = []

        split_text = split_input_text(processed_text)

        for line in split_text:
            if self._should_skip_line(line):
                if contains_invisible_unicode(line):
                    continue
                translated_lines.append(line.strip())
                continue
            if contains_invisible_unicode(line):
                continue
            if len(line) <= 200:
                translated_lines += self._short_sentence_translate(line)
            else:
                translated_lines += self._long_sentence_translate(line)

        translated_text = "\n".join(translated_lines)
        
        # Restore special tags after translation if they were preserved
        if tag_mapping:
            return restore_special_tags(translated_text, tag_mapping)
        return translated_text

    def _short_sentence_translate(self, line: str) -> str:
        translated_lines = []
        needs_translation = True
        if self.source_lang == "en" or line == "$":
            # why is "$" a special line?
            mean_word_judge = is_english(line)
            if not mean_word_judge or line == "$":
                translated_lines.append(line.strip())
                needs_translation = False
            else:
                needs_translation = True
        if needs_translation:
            cleaned_line = self._clean_line(line)
            if cleaned_line:
                translated_line = self._translate(cleaned_line)
                translated_lines.append(translated_line)

        return translated_lines

    def _long_sentence_translate(self, line: str) -> str:
        translated_lines = []
        sentences = re.split(r"(\. |\?)", line.strip())
        for sentence in sentences:
            cleaned_sentence = self._clean_line(sentence)
            if self._should_skip_line(cleaned_sentence):
                translated_lines.append(cleaned_sentence)
                continue
            translated_line = self._translate(cleaned_sentence)
            translated_lines.append(translated_line)

        return translated_lines

    def _should_skip_line(self, line: str) -> bool:
        return (
            line.isspace()
            or line.strip().replace("-", "") == ""
            or len(line) == 0
            or line.replace(".", "") == ""
            or line in {".", "?", ". "}
        )

    def _clean_line(self, line: str) -> str:
        return remove_english_punctuation(line.strip().lower().split())

    def get_text(
        self,
        prompts: List[str],
        reverse_translate_judge: bool = False,
        notify_callback: Callable | None = None,
    ) -> List[str]:
        translated_prompts = []
        prompts_to_process = list(prompts)
        for prompt in prompts_to_process:
            translate_prompt = prompt
            if prompt is not None:
                if reverse_translate_judge:
                    mean_word_judge = is_meaning_string(prompt)
                    if mean_word_judge:
                        translate_prompt = self._get_response(prompt)
                else:
                    translate_prompt = self._get_response(prompt)
            translated_prompts.append(translate_prompt)
            if notify_callback:
                notify_callback()
        return translated_prompts
