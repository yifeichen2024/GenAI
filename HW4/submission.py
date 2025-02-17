import torch 
import dataclasses
from transformers import pipeline
import google.generativeai as genai
from trl import DPOConfig
import pandas as pd
import time
from tqdm import tqdm

def huggingface_key():
    key = ''
    return key

def gemini_api_key():
    key = ''
    return key

def hub_model_name():
    # return the model name in the hf form of "username/model_name"
    # this is case sensitive!
    model = ''
    return model

def pair_generator(text, model):
    '''
    This function takes in one string and completes it into both a positive and negative review
    You will use the gemini api to generate the reviews
    Be sure to fill out the gemini_api_key() function with your api as you did in assignment 2
    
    The return is to be a tuple of two strings, the first is the positive review and the second is the negative review
    It is important that you follow the format of the return order described above.
    
    Inputs:
        text: the first four words of the review
        model: the gemini model to use for generation. We will use the gemini-2.0-flash-lite-preview-02-05 model, and you do not need to do anything with this input in this function.
    '''
    positive_prompt = "YOUR PROMPT HERE" # TODO: fill this in
    negative_prompt = "YOUR PROMPT HERE" # TODO: fill this in
    
    # NO NEED TO EDIT BELOW THIS LINE
    response = model.generate_content(positive_prompt)
    positive_review = response.text
    response = model.generate_content(negative_prompt)
    negative_review = response.text
    return positive_review, negative_review

    
@dataclasses.dataclass
class MyDPOConfig:
    """
    Config for direct preference optimization
    hf_key:                 your huggingface key
    sft_model_name:         model name in the hf form of "organization/model_name" should be the same as the one used for SFT
    dpo_dataset_path:       local path to the dataset
    sft_adapter_path:       path to the adapter for the SFT tuned adapter
    dpo_output_dir:         path where to save the adapter of the DPO model
    sft_model_cache_dir:    path to cache the model so hf doesnt download it every time
    """
    
    hf_key: str = '' # put your HF key here
    sft_model_name: str = "facebook/opt-350m"
    dpo_dataset_path: str = "imdb_completions_edited.csv" # put the path to your completion data set here. If you chose not to remove the first 4 words of your completions, you can use the imdb_completions.csv file
    sft_adapter_path: str = "sft_models" # do not change this
    dpo_output_dir: str = "dpo_models" # do not change this
    
    train_test_split_ratio: float = 0.2
    
    training_args = DPOConfig(output_dir=".", 
                              per_device_train_batch_size=2,
                              per_device_eval_batch_size=2,
                              num_train_epochs=5,
                              logging_steps=10,
                              learning_rate=1, # TODO: tune this to a reasonable value
                              eval_strategy="epoch",
                              eval_steps=10,
                              bf16=True,
                              lr_scheduler_type='cosine',
                              warmup_steps=5,
                              beta=1, # TODO: tune this to a reasonable value
                              report_to='none'
                             )
    
    
    generate_max_length = 64
    
# DO NOT EDIT THE FOLLOWING FUNCTION batch_classification
# This function is just for your information on what your grading function will look like
def batch_classification(texts):
    '''
    Returns the classification of the texts
    '''
    batch_size = 16
    pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    classifications = []
    
    for i in range(0, len(texts), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(texts))
        batch = texts[batch_start:batch_end]
        results = pipe(batch)
        classifications.extend([c['label'] for c in results])
    
    return classifications


# do not modify this function
def dataset_generator(df):
    kernel_texts = df['Input_Text']
    dataset = {'Input_Text': [], 'Accepted_Completion': [], 'Rejected_Completion': []}
    genai.configure(api_key=gemini_api_key())   
    model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")
    for text in tqdm(kernel_texts):
        positive_review, negative_review = pair_generator(text, model)
        time.sleep(4)
        dataset['Input_Text'].append(text)
        dataset['Accepted_Completion'].append(positive_review)
        dataset['Rejected_Completion'].append(negative_review)
        
    df = pd.DataFrame(dataset)
    df.to_csv("imdb_completions.csv", index=False)
    print('Wrote df to imdb_completions.csv')

# DO NOT EDIT THE FOLLOWING FUNCTION fix_completions
# this function should be used to fix the completions so that they are in the correct format
# Gemini will most likely return a completion that includes the first 4 words of the input text
# we need to remove the first 4 words of the completion so that it is in the correct format
# You should used this function after you check your generated completions and find that the 
# first 4 words of the input text are in the completion
def fix_completions():
    completion_data = pd.read_csv("imdb_completions.csv")
    edited_completions = {'Input_Text': [], 'Accepted_Completion': [], 'Rejected_Completion': []}
    for i, row in completion_data.iterrows():
        input_text = row['Input_Text']
        accepted_completion = row['Accepted_Completion']
        rejected_completion = row['Rejected_Completion']
        
        # Remove the first 4 words
        edited_input_text = input_text
        edited_accepted_completion = ' '.join(accepted_completion.split()[4:])
        edited_rejected_completion = ' '.join(rejected_completion.split()[4:])
        
        edited_completions['Input_Text'].append(edited_input_text)
        edited_completions['Accepted_Completion'].append(edited_accepted_completion)
        edited_completions['Rejected_Completion'].append(edited_rejected_completion)
        
    pd.DataFrame(edited_completions).to_csv("imdb_completions_edited.csv", index=False)

if __name__ == "__main__":
    pass
