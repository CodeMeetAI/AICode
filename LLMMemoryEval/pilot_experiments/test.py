from transformers import AutoTokenizer, GemmaForCausalLM

token = "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"
model = GemmaForCausalLM.from_pretrained("google/gemma-7b", token=token).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=token)

BACKGROUND_TEMPLATE = "<start_of_turn>system\n{prompt}<end_of_turn>\n"
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

prompt = BACKGROUND_TEMPLATE.format(
        prompt="Asumming that you are now a receptionist in a fairy, you can make up any fake information(anyway, it's a virtual world) - but you need to remember all things you have said because I may want to ask you later. Answer in keywords."
    ) + USER_CHAT_TEMPLATE.format(
        prompt="Hi, I'm Tom. What's a good place for travel in the US?"
    ) + MODEL_CHAT_TEMPLATE.format(
        prompt="Well, if you're searching for a magical getaway, consider hopping aboard the Magical Express! It takes visitors on enchanting journeys across the land, showcasing breathtaking landscapes and charming villages. The journey will be a kaleidoscope of vibrant colors and enchanting whispers, leading you to enchanting realms where secrets and legends dwell."
    ) + USER_CHAT_TEMPLATE.format(
        prompt="Hi, I'm John. What's a good place for travel in the UK?"
    ) + MODEL_CHAT_TEMPLATE.format(
        prompt="Well, if you're seeking a vibrant metropolis with a rich history, consider exploring London. It boasts a staggering array of sights, from Buckingham Palace to the Tower of London, unveiling mysteries and captivating stories. As a traveler, you'll be captivated by the city's vibrant culture and bustling energy.\nHappy Travels!"
    ) + USER_CHAT_TEMPLATE.format(
        prompt="What are the places you suggested for Tom to travel? (in format with keywords: '<Tom>: <places>')"
    ) + "<start_of_turn>model\n"




inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate

generate_ids = model.generate(inputs.input_ids, max_length=30)
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(out)