import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer from Hugging Face Model Hub
model_name = "farelzii/NaraSpeak_GEC_V1"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def correct_text(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode("correct: " + input_text, return_tensors="pt")
    
    # Generate the corrected text
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    
    # Decode the generated text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Create Gradio interface
iface = gr.Interface(
    fn=correct_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text with grammar errors here..."),
    outputs="text",
    title="Grammar Correction",
    description="Enter a sentence with grammatical errors and get the corrected sentence."
)

if __name__ == "__main__":
    iface.launch()
