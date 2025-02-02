from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

initial_peers = [
    '/ip4/10.0.1.6/tcp/31337/p2p/QmaSwUesrs3QjrGdw5RckdTZkUTdooRVCu6on2mTtJoMeR'
]

# Choose any model available at https://health.petals.dev
# model_name = "bigscience/bloom-560m"
model_name = "deepseek-ai/DeepSeek-R1"

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=initial_peers )

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...