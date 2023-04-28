# cleaner-promts-ChatGPT

👨‍💻 The code imports necessary libraries such as OpenAI, pandas, spacy, gensim, typer, and tabulate.

🔑 The OpenAI API key is set up to authenticate the user.

📝 The spaCy language model is loaded using the en_core_web_sm language model.

🔍 A PorterStemmer stemmer is initialized.

🔠 A custom tokenizer function is defined to tokenize the text and remove stop words and punctuation.

🔮 A function is defined to preprocess the prompt using Gensim's text preprocessing tools.

💡 The is_good_prompt function evaluates the prompt based on various criteria such as relevance, clarity, specificity, creativity, and completeness.

📜 The generate_prompts function generates a single prompt for each topic in the list and returns it if valid.

🧾 The result_list variable stores the validated prompts.

📊 The table_data variable stores the formatted table data for the validated prompts using the tabulate library.

🔤 The typer.echo(table) and print(table) functions display the table data in the console.
