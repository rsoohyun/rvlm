import os

from pydantic import BaseModel
from typing import List
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()


class Parsed(BaseModel):
    result: List[str]
    
PROMPT_DICT = {
    0: "Return list of visual attributes that describes {0}.",
    1: "Return list of visual attributes that describes {0} compared to {1}.",
    2: "Return list of visual attributes that describes {0} in as much detail as possible.",
    3: "Return list of visual attributes that describes {0} in as much detail as possible. Do not include its function, its surroundings, or the environment it usually inhabits.",
    4: "Return list of visual attributes that describes {0} compared to {1}. Do not include its function, its surroundings, or the environment it usually inhabits",
    5: "Return list of visual attributes that describes {0} in as much detail as possible. Each visual attribute should describe a single aspect. Do not include its function, its surroundings, or the environment it usually inhabits.",
    6: "Return list of visual attributes that describes {0} in as much detail as possible. Each visual attribute should describe a single aspect. Do not include its function, its surroundings, or the environment it usually inhabits. For example, visual attributes of a goldfish could be 'long body', 'golden body', 'back fins' and so on.",
}

def generate_desc(class_labels, prompt_id=0, seed=0):
    prompt = PROMPT_DICT[prompt_id]
    
    all_descriptions = {}
    for label in class_labels:
        opp_label = class_labels[0] if label==class_labels[1] else class_labels[1]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful text generation assistant. Given a class label, your goal is to list specific visual attributes."},
                # {"role": "user", "content": "Return list of visual attributes that describes fluffy brown coated dog."},
                # {"role": "assistant", "content": '''"Furry coat", "Four legs", "Tail wagging", "Barking", "Playful behavior", "Snout", "Collar", "Leash", "Walking on all fours", "Wagging tail"'''},
                {"role": "user", "content": prompt.format(label, opp_label) if "{1}" in prompt else prompt.format(label)},
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
