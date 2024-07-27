import os
import sys

import torch
import openai
import numpy as np
import gradio as gr
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    AutoProcessor,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig
)

# Authentication tokens
auth_username = os.environ["AUTH_USERNAME"]
auth_password = os.environ["AUTH_PASSWORD"]
cambridgeltl_access_token = os.environ.get('CAMBRIDGELTL_ACCESS_TOKEN', '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')

# Initialize Llama model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("cambridgeltl/med-alpaca", token=cambridgeltl_access_token)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        "cambridgeltl/med-alpaca",
        use_auth_token=cambridgeltl_access_token,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        "cambridgeltl/med-alpaca",
        use_auth_token=cambridgeltl_access_token,
        device_map={"": device},
        low_cpu_mem_usage=True
    )

model.eval()
if torch.__version__ >= "2" and sys.version_info < (3, 12):
    model = torch.compile(model)

# OpenAI API key
openai.api_key = os.environ.get("OPENAI_TOKEN", None)


def set_openai_api_key(api_key):
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        openai.api_key = api_key


def get_response_from_openai(prompt, model="gpt-3.5-turbo", max_output_tokens=512):
    messages = [{"role": "assistant", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=max_output_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message['content']


torch_dtype = torch.float16 if 'cuda' in device else torch.float32

# Initialize deplot and med-git models
model_deplot = Pix2StructForConditionalGeneration.from_pretrained("google/deplot", torch_dtype=torch_dtype).to(device)
processor_deplot = Pix2StructProcessor.from_pretrained("google/deplot")
model_med_git = AutoModelForCausalLM.from_pretrained('cambridgeltl/med-git-base',
                                                     use_auth_token=cambridgeltl_access_token,
                                                     torch_dtype=torch_dtype).to(device)
processor_med_git = AutoProcessor.from_pretrained('cambridgeltl/med-git-base', use_auth_token=cambridgeltl_access_token)


def evaluate(table, question, llm="med-alpaca", temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=512,
             **kwargs):
    prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context of an uploaded image. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{table}\n\n### Response:\n"
    prompt_no_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"
    prompt = prompt_input if len(table) > 0 else prompt_no_input
    output = "UNKNOWN ERROR"

    if llm == "med-alpaca":
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s).split("### Response:")[1].strip()
    elif llm == "gpt-3.5-turbo":
        try:
            output = get_response_from_openai(prompt)
        except Exception as e:
            output = f"<Error: {str(e)}>"
    else:
        raise RuntimeError(f"No such LLM: {llm}")

    return output


def deplot(image, question, llm):
    inputs = processor_deplot(images=image, text="Generate the underlying data table for the figure below:",
                              return_tensors="pt").to(device, torch_dtype)
    predictions = model_deplot.generate(**inputs, max_new_tokens=512)
    table = processor_deplot.decode(predictions[0], skip_special_tokens=True).replace("<0x0A>", "\n")
    return table


def med_git(image, question, llm):
    inputs = processor_med_git(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values.to(torch_dtype)
    generated_ids = model_med_git.generate(pixel_values=pixel_values, max_length=512)
    captions = processor_med_git.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return captions


def process_document(image, question, llm):
    table = ""
    if image is not None:
        if np.mean(image) >= 128:
            table = deplot(image, question, llm)
        else:
            table = med_git(image, question, llm)
    response = evaluate(table, question, llm=llm)
    return table, response


# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        gr.Markdown(
            """
            <h1 style="text-align: center;"><img src="https://holoware.co/wp-content/uploads/2024/04/Blue-Logo-new.png" style="width:400px"/></h1>
            <h1><center>Holoware Biomedical Language Models</center></h1>
            <p>This is a multi-modal medical foundation model. To use it, simply upload your image and type a question or instruction and click 'submit'.</p>
            """
        )

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            instruction = gr.Textbox(placeholder="Enter your instruction/question...", label="Question/Instruction")
            llm = gr.Dropdown(["med-alpaca", "gpt-3.5-turbo"], label="LLM")
            openai_api_key_textbox = gr.Textbox(
                value='',
                placeholder="Paste your OpenAI API key (sk-...) and hit Enter (if using OpenAI models, otherwise leave empty)",
                show_label=False, lines=1, type='password'
            )
            submit = gr.Button("Submit", variant="primary")

        with gr.Column(scale=2):
            with gr.Accordion("Show intermediate table", open=False):
                output_table = gr.Textbox(lines=8, label="Intermediate Table")
            output_text = gr.Textbox(lines=8, label="Output")

    # Linking actions to components
    openai_api_key_textbox.change(set_openai_api_key, inputs=[openai_api_key_textbox])
    openai_api_key_textbox.submit(set_openai_api_key, inputs=[openai_api_key_textbox])
    submit.click(process_document, inputs=[input_image, instruction, llm], outputs=[output_table, output_text])
    instruction.submit(process_document, inputs=[input_image, instruction, llm], outputs=[output_table, output_text])

demo.queue().launch(auth=(auth_username, auth_password),
                    auth_message="Please enter your username and password to access the Holoware Biomedical Language Models.")
