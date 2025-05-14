# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

# generated_ids = model.generate(**model_inputs, max_length=128)
# tokenizer.batch_decode(generated_ids)[0]


from transformers import pipeline

pipeline_model = pipeline("text-generation")

print(pipeline_model("The secret to baking a good cake is ", max_length=50))