import tiktoken

def test_tokenization(string):
    encoding = tiktoken.get_encoding("o200k_base")
    encoded_string = encoding.encode(string)
    #token_bytes = [encoding.decode_single_token_bytes(token) for token in encoded_string]
    token_bytes = encoding.decode_tokens_bytes(encoded_string)

    print(f"No of tokens: {len(encoded_string)}")
    print(f"Encoded string: {encoded_string}")
    print(f"Decoded string: {token_bytes}")

#test_string = "€  😄  你"

#test_string_array = ["hello", "Hello", "HELLO"]
# test_string_array = ["ChatGPT", "chatgpt", "chat gpt"]
# for value in test_string_array:
#     test_tokenization(value)
#     print("\n")

test_string_in_english = "Sunlight, still soft and golden, filters through the leaves as I begin my morning walk, a daily ritual that grounds my day before the world truly awakens. The air, crisp and carrying the scent of damp earth and blooming jasmine, feels clean and revitalizing, a stark contrast to the city's later bustle."
test_string_in_hindi = "सुबह की सैर शुरू करते ही सूरज की कोमल और सुनहरी किरणें पत्तियों से छनकर आती हैं। यह एक दैनिक दिनचर्या है जो दुनिया के जागने से पहले मेरे दिन को एक नई दिशा देती है। हवा ताज़ी है और उसमें नम मिट्टी और खिलते चमेली की खुशबू घुली हुई है, जो स्वच्छ और स्फूर्तिदायक महसूस होती है, शहर की बाद की भागदौड़ से बिलकुल अलग।"

test_tokenization(test_string_in_english)
print("\n")
test_tokenization(test_string_in_hindi)