from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load from Render persistent disk
model_dir_as_to_en = "/app/models/as_to_en"
model_dir_en_to_as = "/app/models/en_to_as"

tokenizer_as_to_en = AutoTokenizer.from_pretrained(model_dir_as_to_en)
model_as_to_en = AutoModelForSeq2SeqLM.from_pretrained(model_dir_as_to_en)

tokenizer_en_to_as = AutoTokenizer.from_pretrained(model_dir_en_to_as)
model_en_to_as = AutoModelForSeq2SeqLM.from_pretrained(model_dir_en_to_as)

@app.post("/translate_as_to_en")
async def translate_as_to_en(data: dict):
    text = data["text"]
    tokens = tokenizer_as_to_en(text, return_tensors="pt")
    output = model_as_to_en.generate(**tokens)
    translation = tokenizer_as_to_en.decode(output[0], skip_special_tokens=True)
    return {"translation": translation}

@app.post("/translate_en_to_as")
async def translate_en_to_as(data: dict):
    text = data["text"]
    tokens = tokenizer_en_to_as(text, return_tensors="pt")
    output = model_en_to_as.generate(**tokens)
    translation = tokenizer_en_to_as.decode(output[0], skip_special_tokens=True)
    return {"translation": translation}
