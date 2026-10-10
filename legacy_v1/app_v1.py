import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Socratic Math Tutor",
    page_icon="🎓",
    layout="centered"
)

SYSTEM_PROMPT = (
    "You are an expert Socratic Math Tutor. Guide the student step-by-step using "
    "calibrated hints and questions. Do NOT reveal the final numerical answer directly."
)

@st.cache_resource
def load_model():
    """Loads base model and attaches the fine-tuned LoRA adapters."""
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./socratic_model/socratic_qwen_lora"
    
    # Detect device (CUDA GPU -> Apple Silicon MPS -> CPU)
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True
    )
    if device == "mps":
        base_model = base_model.to(device)

    # Attach LoRA adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer, device

# Initialize UI Header
st.title("🎓 Socratic Math Tutor")
st.caption("A parameter-efficient fine-tuned assistant that guides learning rather than spoon-feeding answers.")

# Load model with loading indicator
with st.spinner("Loading Socratic Tutor weights..."):
    model, tokenizer, device = load_model()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Display Existing Chat Messages (excluding system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User Input Handling
if prompt := st.chat_input("Ask a math question or share your current step..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare chat template input (Request dict explicitly for safety)
    formatted_input = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True 
    ).to(device)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=formatted_input["input_ids"],
                    attention_mask=formatted_input["attention_mask"],
                    max_new_tokens=160,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode only the generated completion
            new_tokens = outputs[0][formatted_input["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            st.markdown(response)

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar controls
with st.sidebar:
    st.header("Session Settings")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()