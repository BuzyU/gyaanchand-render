from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

# 🧠 Load your model from Hugging Face
model_name = "UmerZingu/gyaanchand-checkpoint-10750"

print("🔄 Loading Gyaanchand model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

def gyaanchand_chat(message, history):
    # Combine chat history for conversational context
    context = ""
    for user, bot in history:
        context += f"User: {user}\nGyaanchand: {bot}\n"
    context += f"User: {message}\nGyaanchand:"

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Gyaanchand:")[-1].strip()
    return response

chat_ui = gr.ChatInterface(
    fn=gyaanchand_chat,
    title="🧠 Gyaanchand - AI Assistant",
    description="Chat naturally with Gyaanchand, your custom fine-tuned model!",
    theme="soft"
)

if __name__ == "__main__":
    chat_ui.launch(server_name="0.0.0.0", server_port=7860)
