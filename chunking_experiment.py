# chunking_experiment.py
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 如果没有安装 nltk 或 spacy，先装一下
# pip install nltk spacy
# python -m spacy download zh_core_web_sm


def recursive_split(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    return splitter.split_text(text)

def fixed_char_split(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def token_split(text, chunk_size=100, chunk_overlap=10):
    from langchain_text_splitters import TokenTextSplitter
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def nltk_sentence_split(text, chunk_size=5):  # chunk_size=句子数量
    splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=1)
    return splitter.split_text(text)

def markdown_header_split(markdown_text):
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(markdown_text)