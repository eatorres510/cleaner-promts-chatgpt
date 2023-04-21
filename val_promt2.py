import openai
import pandas as pd
import spacy
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.porter import PorterStemmer
import typer
from tabulate import tabulate

app = typer.Typer()

# Set up the OpenAI API credentials
openai.api_key = "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"

# Load the language model and language tool
nlp = spacy.load("en_core_web_sm")

# Initialize a stemmer
porter_stemmer = PorterStemmer()

# Define a custom tokenizer function
def custom_tokenizer(text):
    """Tokenize the text and remove stop words and punctuation"""
    return [porter_stemmer.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

def preprocess_prompt(prompt):
    """Preprocess the prompt using Gensim's text preprocessing tools"""
    tokens = custom_tokenizer(prompt)
    return " ".join(tokens)

def is_good_prompt(prompt):
    # Evaluate the prompt based on the criteria
    relevance_score = 0
    clarity_score = 0
    specificity_score = 0
    creativity_score = 0
    completeness_score = 0
    
    # You can adjust the weights and thresholds to reflect your evaluation criteria and priorities
    relevance_weight = 0.2
    relevance_threshold = 0.7
    
    clarity_weight = 0.2
    clarity_threshold = 0.7
    
    specificity_weight = 0.2
    specificity_threshold = 0.7
    
    creativity_weight = 0.2
    creativity_threshold = 0.7
    
    completeness_weight = 0.2
    completeness_threshold = 0.7
    
    # Evaluate relevance
    if "python" in prompt.lower():
        relevance_score += 1
        
    # Evaluate clarity
    if "cleaner" in prompt.lower() and "prompts" in prompt.lower() and "python" in prompt.lower():
        clarity_score += 1
        
    # Evaluate specificity
    if "cleaner" in prompt.lower() and "prompts" in prompt.lower() and "python" in prompt.lower():
        specificity_score += 1
        
    # Evaluate creativity
    if "alternative" in prompt.lower() or "innovative" in prompt.lower():
        creativity_score += 1
        
    # Evaluate completeness
    if "syntax" in prompt.lower() and "semantics" in prompt.lower() and "best practices" in prompt.lower():
        completeness_score += 1
    
    # Compute the overall score
    overall_score = (relevance_score * relevance_weight + 
                     clarity_score * clarity_weight + 
                     specificity_score * specificity_weight + 
                     creativity_score * creativity_weight + 
                     completeness_score * completeness_weight)
    
    # Determine whether the prompt is good or bad
    if overall_score >= 1 and (relevance_score >= relevance_threshold and
                               clarity_score >= clarity_threshold and
                               specificity_score >= specificity_threshold and
                               creativity_score >= creativity_threshold and
                               completeness_score >= completeness_threshold):
        return "Good prompt"
    else:
        return "Bad prompt"

    
def generate_prompts(topics):
    """Generate a single prompt for each topic in the list and return it if valid."""
    # Initialize an empty list to store validated prompts
    validated_prompts = []
    
    # Loop over each topic in the list
    for topic in topics:
        # Generate a single prompt for the current topic
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=topic,
            max_tokens=100,
            n=10,
            stop=None,
            temperature=0.7,
        )
        # Check if the generated prompt is valid
        prompt = response.choices[0].text.strip()
        preprocessed_prompt = preprocess_prompt(prompt)
        if is_good_prompt(preprocessed_prompt):
            validated_prompts.append(preprocessed_prompt)
    
    return validated_prompts

topics = ["machine learning", "data science", "natural language processing"]
prompts = generate_prompts(topics)

result_list = prompts
headers = ["Prompts"]
table_data = [[str(item)] for i, item in enumerate(result_list)]
table = tabulate(table_data, headers=headers, tablefmt="pretty")
typer.echo(table)
print(table)