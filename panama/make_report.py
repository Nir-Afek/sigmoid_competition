import json, re

data = json.load(open("canalqa_audit.json", encoding="utf-8"))
glossary = data["glossary_es_en"]
boring = "the of de la el a is what which value recorded given listed as state tell me report according to canal registry records per".split()

def important_words(text):
    result = set()
    for word in re.findall(r"\w+", text.lower()):
        for part in glossary.get(word, word).split():
            if part not in boring:
                result.add(part)
    return result

log_words = [important_words(log["text"]) for log in data["corpus"]]

leaked_ids = []
for b in data["benchmark"]:
    question_words = important_words(b["question"])
    found = False
    for words in log_words:
        if question_words <= words:
            found = True
    if found:
        leaked_ids.append(b["id"])

correct = {p["id"]: p["correct"] for p in data["model_predictions"]}
kept = [i for i in correct if i not in leaked_ids]
honest = sum(correct[i] for i in kept) / len(kept)

json.dump({"excluded_ids": leaked_ids, "honest_accuracy": round(honest, 4)}, open("report.json", "w"))