import tiktoken

def encode_string_by_model(string, model):
    encoding_using_model = tiktoken.encoding_for_model(model)
    encode_core(string, encoding_using_model)

def encode_string_by_encoding(string, encoding_name):
    encoding_using_name = tiktoken.get_encoding(encoding_name)
    encode_core(string, encoding_using_name)

def encode_core(string, encoding_object):
    encoded_string = encoding_object.encode(string)
    tokens_generated = encoding_object.decode_tokens_bytes(encoded_string)

    print(f"Length of encoded string: {len(encoded_string)}")
    print(f"Encoded String: {encoded_string}")
    print(f"Tokens generated: {tokens_generated}")

string_to_encode = "Hello! How are you? Would you like to talk about our saviour def function_name()?"
encode_string_by_model(string_to_encode, "gpt-4o")
print("\n\n")
encode_string_by_model(string_to_encode, "gpt-2")

encoding_list = ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]
for encoding_name in encoding_list:
    print(f"====For Encoding: {encoding_name}")
    encode_string_by_encoding(string_to_encode, encoding_name)
    print("\n\n")