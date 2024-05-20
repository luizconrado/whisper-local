import ollama

# Assuming ollama.list() returns a dictionary with a 'models' key containing the list of model dictionaries
models = ollama.list().values()  # This retrieves the list of models

for model in models:
    for item in model:
        print(item['name'])