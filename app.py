from transformers import pipeline

import gradio as gr

ner_pipeline = pipeline("token-classification", model="the-neural-networker/xlm-roberta-base-finetuned-panx-all")

examples = [
    "Does Chicago have any stores and does Joe live here?",
]

def ner(text):
    output = ner_pipeline(text)
    return {"text": text, "entities": output}    

demo = gr.Interface(ner,
             gr.Textbox(placeholder="Enter sentence (English, Hindi, Telugu, Tamil) here..."), 
             gr.HighlightedText(),
             examples=examples)

demo.launch()
