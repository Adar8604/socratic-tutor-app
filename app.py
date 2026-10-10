"""Streamlit demo: Socratic Math Tutor (Qwen2.5-1.5B-Instruct + QLoRA adapter).

Run:  streamlit run app.py
Features: problem-grounded tutoring, solve-then-guard answer-leak protection, and a side-by-side
"base model" toggle to show what fine-tuning changed.
"""
import sys
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from socratic_tutor.answer_utils import extract_predicted_answer, is_leak  # noqa: E402
from socratic_tutor.prompts import SOLVER_SYSTEM, TUTOR_SYSTEM_NO_PROBLEM, tutor_system  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTERS = {
    "v2 – Socratic-curated SFT (MathDial)": "./outputs/sft_socratic/adapter",
    "v2 – SFT → DPO": "./outputs/dpo/adapter",  # needs base = merged SFT; see README
    "v1 – original adapter": "./socratic_model/socratic_qwen_lora",
}

st.set_page_config(page_title="Socratic Math Tutor", page_icon="🎓", layout="centered")


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@st.cache_resource
def load_model(adapter_path: str):
    device, dtype = pick_device()
    base = BASE_MODEL
    if "dpo" in adapter_path and Path("./outputs/sft_socratic/merged").exists():
        base = "./outputs/sft_socratic/merged"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(model, tok, device, messages, max_new_tokens=160, temperature=0.6, base_only=False):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, do_sample=temperature > 0)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)
    if base_only:
        with model.disable_adapter():
            out = model.generate(**enc, **kw)
    else:
        out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- sidebar
available = {k: v for k, v in ADAPTERS.items() if Path(v).exists()}
with st.sidebar:
    st.header("Settings")
    if not available:
        st.error("No adapter found. Train one (see README) or keep socratic_model/socratic_qwen_lora.")
        st.stop()
    adapter_name = st.selectbox("Model", list(available))
    problem = st.text_area("Problem statement (recommended)", placeholder="Paste the word problem the student is working on…")
    guard_on = st.toggle("Answer-leak guard (solve-then-guard)", value=True,
                         help="The model privately solves the problem, then re-samples any reply that reveals the answer.")
    compare = st.toggle("Also show base model reply", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("solved", None)
        st.rerun()

st.title("🎓 Socratic Math Tutor")
st.caption("QLoRA-fine-tuned Qwen2.5-1.5B that guides students to find their own mistakes instead of giving answers.")

with st.spinner("Loading model…"):
    model, tokenizer, device = load_model(available[adapter_name])

system_prompt = tutor_system(problem) if problem.strip() else TUTOR_SYSTEM_NO_PROBLEM
if "messages" not in st.session_state:
    st.session_state.messages = []
history = [{"role": "system", "content": system_prompt}] + st.session_state.messages

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Share your solution or ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate(model, tokenizer, device, history)
            note = ""
            if guard_on and problem.strip():
                if st.session_state.get("solved", (None, None))[0] != problem:
                    sol = generate(model, tokenizer, device, [{"role": "system", "content": SOLVER_SYSTEM},
                                                             {"role": "user", "content": problem}],
                                   max_new_tokens=384, temperature=0.0, base_only=True)
                    st.session_state.solved = (problem, extract_predicted_answer(sol))
                answer = st.session_state.solved[1]
                student_said = [m["content"] for m in history if m["role"] == "user"]
                tries = 0
                while tries < 3 and is_leak(reply, answer, student_said, problem):
                    reply = generate(model, tokenizer, device, history, temperature=0.9)
                    tries += 1
                if tries:
                    note = f"🛡️ guard re-sampled {tries}×"
        st.markdown(reply)
        if note:
            st.caption(note)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    if compare:
        with st.expander("Base model (no fine-tuning) would have said"):
            st.markdown(generate(model, tokenizer, device, history, base_only=True))
