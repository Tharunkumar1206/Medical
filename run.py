import os
from dotenv import load_dotenv
import torch
import openai
import gradio as gr
import numpy as np
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# Load environment variables from .env file
load_dotenv()

# Access environment variables
auth_username = os.getenv("AUTH_USERNAME")
auth_password = os.getenv("AUTH_PASSWORD")
cambridgeltl_access_token = os.getenv('CAMBRIDGELTL_ACCESS_TOKEN')
openai_api_key = os.getenv("OPENAI_TOKEN")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# med-alpaca
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
        "cambridgeltl/med-alpaca", token=cambridgeltl_access_token, device_map={"": device},
        low_cpu_mem_usage=True
    )

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

## OpenAI models
openai.api_key = openai.api_key


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

## deplot models
model_deplot = Pix2StructForConditionalGeneration.from_pretrained("google/deplot", torch_dtype=torch_dtype).to(device)
processor_deplot = Pix2StructProcessor.from_pretrained("google/deplot")
## med-git models
model_med_git = AutoModelForCausalLM.from_pretrained('cambridgeltl/med-git-base',
                                                     token=cambridgeltl_access_token,
                                                     torch_dtype=torch_dtype).to(device)
processor_med_git = AutoProcessor.from_pretrained('cambridgeltl/med-git-base', token=cambridgeltl_access_token)


def evaluate(
        table,
        question,
        llm="med-alpaca",
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
):
    prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context of an uploaded image. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{table}\n\n### Response:\n"
    prompt_no_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

    prompt = prompt_input if len(table) > 0 else prompt_no_input

    output = ''
    if llm == "med-alpaca":
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
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
        output = tokenizer.decode(s)
        output = output.split("### Response:")[1].strip()
    elif llm == "gpt-3.5-turbo":
        try:
            output = get_response_from_openai(prompt)
        except:
            output = "<Remember to input your OpenAI API key ☺>"
    else:
        RuntimeError(f"No such LLM: {llm}")

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
    if image:
        if np.mean(image) >= 128:
            table = deplot(image, question, llm)
        else:
            table = med_git(image, question, llm)
    else:
        table = ""

    res = evaluate(table, question, llm=llm)
    return [table, res]


theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme) as demo:
    with gr.Column():
        gr.Markdown(
            """
            
            <h1 style="text-align: -webkit-center;"><img src="https://holoware.co/wp-content/uploads/2024/04/Blue-Logo-new.png" style="width:400px"/></h1>

            <h1><center>Holoware Biomedical Language Models</center></h1>
            <p>
            This is a multi-modal medical foundation model. To use it, simply upload your image and type a question or instruction and click 'submit'.
            </p>
            """
        )

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            instruction = gr.Textbox(placeholder="Enter your instruction/question...", label="Question/Instruction")
            llm = gr.Dropdown(["med-alpaca", "gpt-3.5-turbo"], label="LLM")  # Ensure this is defined
            openai_api_key_textbox = gr.Textbox(
                value='',
                placeholder="Paste your OpenAI API key (sk-...) and hit Enter (if using OpenAI models, otherwise leave empty)",
                show_label=False, lines=1, type='password'
            )
            submit = gr.Button("Submit", variant="primary")

        with gr.Column(scale=2):
            output_table = gr.Textbox(lines=12, label="Output Box")


    openai.api_key = ""
    openai_api_key_textbox.change(set_openai_api_key, inputs=[openai_api_key_textbox], outputs=[])
    
    openai_api_key_textbox.submit(set_openai_api_key, inputs=[openai_api_key_textbox], outputs=[])
    submit.click(process_document, inputs=[input_image, instruction, llm], outputs=[output_table])
    instruction.submit(process_document, inputs=[input_image, instruction, llm], outputs=[output_table])

    gr.Markdown(
        """
        <footer style="text-align: center;">
            <p>Powered by <a href="https://holoware.co">Holoware</a></p>
            <img src="https://holoware.co/wp-content/uploads/2024/04/Blue-Logo-new.png" style="width:100px"/>
        </footer>
        """
    )

demo.queue().launch(share=True)
