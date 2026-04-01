import re
from datasets import load_dataset
from config import config


def parse_row(text):
    patterns = {
        "symptoms": r"\[SYMPTOMS\]:\s*'(.*?)'",
        "causes": r"\[CAUSES\]:\s*'(.*?)'",
        "actions": r"\[ACTIONS\]:\s*'(.*?)'",
    }

    result = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        result[key] = match.group(1).strip() if match else None

    return result


def create_chunks(dataset):
    chunks = []

    for i, row in enumerate(dataset):
        parsed = parse_row(row["text"])
        
        for tag, content in parsed.items():
            if content is None:
                continue

            chunks.append({
                "scenario_id": i,
                "tag": tag,
                "text": content
            })

    return chunks


def load_data():
    dataset = load_dataset(config["dataset"]["name"])
    split = dataset[config["dataset"]["split"]]
    return create_chunks(split)