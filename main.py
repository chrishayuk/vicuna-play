import copy
from llama_cpp import Llama

# load the model
print("Loading model...")
#llm = Llama(model_path="./models/ggml-vicuna-13b-4bit-rev1.bin")
llm = Llama(model_path="./models/ggml2-alpaca-7b-q4.bin")
print("Model loaded.")

# run the model
print("Running model...")
stream = llm(
    "Question: Who is ada lovelace? Answer:",
    max_tokens=100,
    stop=["\n", "Question:", "Q:"],
    stream=True,
)

# print the output
for output in stream:
    # convert the output to json
    completionFragment = copy.deepcopy(output)
    print(completionFragment["choices"][0]["text"])