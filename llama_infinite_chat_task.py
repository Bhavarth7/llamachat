
!pip install -r requirements.txt

import json
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline)

config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token

"""**Load the configuration and set up the bites and bites config**"""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

"""**Setup the text generation pipeline**

"""

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)



def get_response(prompt):
  sequences = text_generator(prompt)
  gen_text = sequences[0]["generated_text"]
  return gen_text

prompt = "What is Machine Learning?"

llama3_response = get_response(prompt)

print(llama3_response)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(inputs.input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


print("Chat with Llama-3 (type 'exit' to stop):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print(f"Llama-3: {response}")