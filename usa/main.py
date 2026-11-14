import json, os, tiktoken

inp, out = ("test-input.json", "test-output.json") if os.path.exists("test-input.json") \
    else ("train-input.json", "train-output.json")

data = json.load(open(inp, encoding="utf-8-sig"))
text = data["calibration"]["text"]
cal_ids = data["calibration"]["token_ids"]
challenge = data["challenge"]["token_ids"]

for name in tiktoken.list_encoding_names():
    enc = tiktoken.get_encoding(name)
    if enc.encode(text, allowed_special="all") == cal_ids:
        result = {"tokenizer_id": name, "decoded_message": enc.decode(challenge)}
        json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(result)
        break