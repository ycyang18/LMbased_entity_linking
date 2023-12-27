
import os
import re
import json
import torch
import openai
import weaviate
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from weaviate.embedded import EmbeddedOptions
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.configs import *

def initialize_weaviate_client():
    client = weaviate.Client(
        embedded_options=EmbeddedOptions(),
        additional_headers = {
            "X-OpenAI-Api-Key": OPENAI_API_KEY_FOR_WEAVIATE #OPENAI_API_KEY 
        }
    )
    return client

def load_jsonl_to_df(file_path):
    articles = []
    with open(file_path, 'r') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles

def load_json(file_path):
    with open(file_path, 'r') as file:
        dic = json.load(file)
    return dic
