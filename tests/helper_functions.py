from __future__ import annotations

import os
import regex as re
from typing import IO, BinaryIO, Iterable, Optional, Type, List
from collections import defaultdict
import json
import numpy.typing as npt
import math
# pyrefly: ignore [missing-import]
import torch
import heapq

# BPE_TRAINING, GET_TOKENIZER
def build_pretokenizer_pattern(special_tokens: list[str]) -> re.Pattern:
    """Build the pre-tokenizer pattern - use in both training and encoding."""
    parts = []
    
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        parts.append(f"(?:{special_pattern})")
    
    parts.extend([
        r"'s|'t|'re|'ve|'m|'ll|'d",
        r"\ ?\p{L}+",
        r"\ ?\p{N}+",
        r"\ ?[^\s\p{L}\p{N}]+",
        r"\s+(?!\S)",
        r"\s+"
    ])
    
    return re.compile("|".join(parts))

# GET_TOKENIZER
def pre_tokenizer_encode(text: str, special_tokens: list[str]) -> list[bytes]:
    """Split text into tokens, preserving special tokens as atomic units."""
    if not special_tokens:
        # No special tokens - use regex directly
        pattern = build_pretokenizer_pattern([])
        out = []
        for m in pattern.finditer(text):
            token = m.group(0)
            out.extend([bytes([b]) for b in token.encode()])
        return out
    
    # Split on special tokens first
    special_set = set(special_tokens)
    
    # Sort special tokens by length (longest first) for correct splitting
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    
    # Build a regex to split on special tokens
    split_pattern = "|".join(re.escape(tok) for tok in sorted_special)
    split_regex = re.compile(f"({split_pattern})")
    
    # Split text, keeping the delimiters (special tokens)
    parts = split_regex.split(text)
    
    # Now process each part
    pattern = build_pretokenizer_pattern([])  # Pattern without special tokens
    out = []
    
    for part in parts:
        if not part:  # Skip empty strings
            continue
        
        if part in special_set:
            # This is a special token - keep it atomic
            out.append(part.encode())
        else:
            # Regular text - apply regex tokenization
            for m in pattern.finditer(part):
                token = m.group(0)
                out.extend([bytes([b]) for b in token.encode()])
    
    return out

# BPE_TRAINING
def pre_tokenizer_train(text: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    pattern = build_pretokenizer_pattern(special_tokens)
    tokens = [m.group(0) for m in pattern.finditer(text)]

    pretokens = defaultdict(int)
    special_set = set(special_tokens)

    for tok in tokens: # O(n)
        if tok in special_set:
            pretokens[(tok.encode("utf-8"),)] += 1 # If speacial toke, then take as one entity, like <PAD> is one not <, P, A, D, >
        else:
            pretokens[tuple(bytes([b]) for b in tok.encode("utf-8"))] += 1 # Just split and add, like Hi is b'H' and b'i'

    return pretokens # Tuple wih freq

# LEARNING RATE SCHEDULING
def linear_warmup(t: int, alpha_max: float, T_w: int) -> float:
    return (t / T_w) * alpha_max

def cosine_annealing(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    cos_term = math.cos(((t - T_w) / (T_c - T_w)) * math.pi)
    return alpha_min + 0.5 * (1.0 + cos_term) * (alpha_max - alpha_min)

def post_annealing(alpha_min: float) -> float:
    return alpha_min