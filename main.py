import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import nltk

# Ensure nltk tokenizer is downloaded
nltk.download('punkt')

# Load Cybersecurity Dataset
cybersecurity_data = [
    {"text": "Always update your software to the latest version to avoid known vulnerabilities."},
    {"text": "Use strong, unique passwords and enable multi-factor authentication for better security."},
    {"text": "Never run unverified scripts with sudo/root privileges."},
    {"text": "Check your system logs for any unusual activity to detect potential security breaches."},
    {"text": "Perform regular vulnerability scans using tools like Nmap, Nessus, or OpenVAS."},
    {"text": "Ensure your firewall and antivirus are always active and updated."},
    {"text": "Use a VPN to encrypt your internet traffic and prevent tracking."},
    {"text": "When developing cybersecurity tools, avoid hardcoding sensitive credentials."},
    {"text": "Phishing emails often have urgent language, unexpected attachments, and suspicious links."},
]

# Convert to Dataset format
dataset = Dataset.from_list(cybersecurity_data)

# Model & Tokenizer
model_name = "distilgpt2"  # Lightweight GPT model for local use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./cybersec_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masked language modeling
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("./cybersec_model")
tokenizer.save_pretrained("./cybersec_model")

print("Training Complete! Model Saved as 'cybersec_model'")

def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=150,  # Increase response length
        do_sample=True,  # Enable sampling
        top_k=50,  # Top-K sampling (reduce repetitive answers)
        top_p=0.95,  # Top-P nucleus sampling (keeps only high-probability words)
        temperature=0.7,  # Adds randomness to responses
        repetition_penalty=1.2  # Avoids repeating words
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example Queries
print("\nCybersecurity Assistant Ready!")
while True:
    query = input("\nAsk me a cybersecurity question: ")
    if query.lower() in ["exit", "quit"]:
        break
    print("\nResponse:", query_model(query))
