from transformers import pipeline

import gradio as gr

ner_pipeline = pipeline("token-classification", model="the-neural-networker/xlm-roberta-base-finetuned-panx-all")

examples = [
    "Does Chicago have any stores and does Joe live here?",
]

def ner(text):
    output = ner_pipeline(text)
    return {"text": text, "entities": output}    


if __name__ == "__main__":
    # define app features and run
    title = "Multilingual Language Recognition Demo"
    description = "<p style='text-align: center'>Gradio demo for a Multilingual Language Recognition model, viz., XLM-RoBERTa finetuned on the XTREME dataset's English, Hindi, Telugu, and Tamil languages. To use it, type your text, or click one of the examples to load them. Since this demo is run on CPU only, please allow additional time for processing. </p>"
    article = "<p style='text-align: center'><a href='https://github.com/the-neural-networker/multi-lingual-language-recognition'>Github Repo</a></p>"
    css = "#0 {object-fit: contain;} #1 {object-fit: contain;}"
    demo = gr.Interface(fn=ner, 
                        title=title, 
                        description=description,
                        article=article,
                        inputs=gr.Textbox(placeholder="Enter sentence (English, Hindi, Telugu, Tamil) here..."), 
                        outputs=gr.HighlightedText(),
                        css=css, 
                        examples=examples, 
                        cache_examples=True,
                        allow_flagging='never')
    demo.launch()
