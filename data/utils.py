import os

from pydantic import BaseModel
from typing import List
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()


class Parsed(BaseModel):
    result: List[str]

def generate_desc(class_labels, seed=0):
    prompt = "Return list of visual attributes that describes {0}."
    
    all_descriptions = {}
    for label in class_labels:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful text generation assistant. Given a class label, your goal is to list specific visual attributes."},
                # {"role": "user", "content": "Return list of visual attributes that describes fluffy brown coated dog."},
                # {"role": "assistant", "content": '''"Furry coat", "Four legs", "Tail wagging", "Barking", "Playful behavior", "Snout", "Collar", "Leash", "Walking on all fours", "Wagging tail"'''},
                {"role": "user", "content": prompt.format(label)},
            ],
            response_format=Parsed,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            seed=seed,
        )
        response = completion.choices[0].message.parsed.result
        all_descriptions[label] = response
    return all_descriptions
