from transformers import GPT2Tokenizer, GPT2LMHeadModel 
import torch
import gradio as gr

tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/gpt2-fake-news')
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/gpt2-fake-news')
model.to("cuda" if torch.cuda.is_available() else "cpu")

def genearte_fake_news(prompt):
  inputs = tokenizer(prompt,return_tensors='pt')
  inputs = {k:v.to(model.device) for k,v in inputs.items()}

  outputs = model.generate(
      **inputs,
      max_length=200,
      do_sample=True,
      top_k=50,
      top_p=0.95,
      temperature=0.8,
      pad_token_id=tokenizer.eos_token_id,
      early_stopping = False
  )
  return tokenizer.decode(outputs[0],skip_special_tokens=True)

iface = gr.Interface(
    fn=genearte_fake_news,
    inputs=gr.Textbox(lines=2, placeholder="Enter a prompt, e.g., 'The government announced...'"),
    outputs="text",
    title="Fake News Generator",
    description="Enter a prompt to generate fake news using GPT-2"
)

iface.launch(share=True)
