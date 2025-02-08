import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments 
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd

# There are a total of 5 TODOs in this file.

@dataclass
class SFTConfig:
    sft_model_name: str = 'facebook/opt-350m'
    sft_dataset_path: str = 'train.csv'
    sft_model_cache_dir: str = 'cache'
    sft_output_dir: str = '.'
    hf_key: str = ''

    peft_config = LoraConfig(
        r=4, # TODO: play with this number 
        lora_alpha=32, # TODO: play with this number 
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM" # TODO: you need to figure this out. HINT https://github.com/huggingface/peft/blob/3d2bf9a8b261ed2960f26e61246cf0aa624a6115/src/peft/utils/peft_types.py#L67
    )
        
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing =False,
        max_grad_norm= 0.3,
        num_train_epochs=3, # TODO: play with this number 
        save_steps= 100,
        learning_rate=1e-4, # TODO: play with this number 
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir='./sft_models',
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="none",
    )
    
    generate_max_length: int = 64

#  There are not more TODOs after this point.

def load_model(config: SFTConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def load_dataset(dataset: pd.DataFrame, config: SFTConfig):
    dataset = Dataset.from_pandas(dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        text = f"### Question: {example['Input_Text']}\n ### Answer: {example['Solution']}"
        return {'formatted_text': text}
    
    def tokenize_func(example):
        tokenized_sample = tokenizer(example['formatted_text'], padding=True, truncation=True, return_special_tokens_mask=True)
        return tokenized_sample
    
    formatted_dataset = dataset.map(formatting_prompts_func)
    tokenized_dataset = formatted_dataset.map(tokenize_func, remove_columns=['Input_Text', 'Solution', 'formatted_text',])
    
    collator = DataCollatorForCompletionOnlyLM(" ### Answer:", tokenizer=tokenizer, return_tensors="pt")
    return tokenized_dataset, collator

def load_adapter(config: SFTConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    model = PeftModel.from_pretrained(model, config.sft_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_inference(config: SFTConfig, input_text: str):
    model, tokenizer = load_adapter(config)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def run_inference_with_model(model, tokenizer, input_text, config: SFTConfig):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def run_inference_from_hub(hub_model_name: str, input_text: str, config: SFTConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 token=config.hf_key,
                                                 device_map="auto",
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    model = PeftModel.from_pretrained(model, hub_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def train(config: SFTConfig, dataset: pd.DataFrame):
    model, tokenizer = load_model(config)
    tokenized_dataset, collator = load_dataset(dataset, config)
    model = get_peft_model(model, config.peft_config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=config.training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    return trainer.model

if __name__ == "__main__":
    config = SFTConfig()
    train(config)