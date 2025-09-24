import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM
import gradio as gr

MODEL_NAME = "Qwen/Qwen2-7B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def gen_doc(doc_type, name, email, phone, location, summary, skills, experience, education, target_role, company, interest):
    user_prompt = gen_user_prompt(doc_type, name, email, phone, location, summary, skills, experience, education, target_role, company, interest)
    messages=[
        {"role": "system", "content": "You are a helpful assistant that assist in writing professional and excellent resume and CV"},
        {"role" : "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def gen_user_prompt( doc_type, name, email, phone, location, summary, skills, experience, education, target_role, company, interest):
    if doc_type =="Resume":
        return f"""
            create a professional resume based on the following informations
            Name: {name}
            Email: {email}
            Phone: {phone}
            Location: {location}
            Summary: {summary}
            Skills: {skills}
            Experience: {experience}
            Education: {education}
        """
    else :
        return f"""
            create a professional cover letter based on the following informations
            Name: {name}
            Email: {email}
            Phone: {phone}
            Location: {location}
            Summary: {summary}
            Skills: {skills}
            Experience: {experience}
            Education: {education}
        """
    

with gr.Blocks() as demo:
    gr.Markdown("Resume and Cover Letter Generator")
    doc_type = gr.Radio(["Resume", "Cover Letter"], label="Document Type", value="Resume")
    name = gr.TextBox(label="Full Name")
    email = gr.TextBox(label="Email")
    phone = gr.TextBox(label="Phone Number")
    location = gr.TextBox(label="Location")
    summary = gr.TextBox(label="Professional Summary", lines=4)
    skills = gr.Textbox(label="Skills (comma-separated)")
    experience = gr.Textbox(label="Work Experience", lines=4)
    education = gr.Textbox(label="Education")
    target_role = gr.Textbox(label="Target Role")
    company = gr.Textbox(label="Company Name")
    interest = gr.Textbox(label="Why you're interested in the role", lines=3)

    output = gr.Textbox(label="Generated Output", lines=20)

    btn = gr.Button("Generate Document")

    btn.click(
        fn=gen_doc,
        inputs=[doc_type, name, email, phone, location, summary, skills, experience, education, target_role, company, interest],
        outputs=output
    )
demo.launch()