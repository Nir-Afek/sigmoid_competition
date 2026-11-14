import json
import re

docs = json.load(open("test-input.json", encoding="utf-8"))["documents"]

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre"

PATTERNS = [
    (r"\b[A-Z]{2}\d{2}[\d ]{15,30}\d\b", "[IBAN]"),
    (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "[DOB]"),
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[DOB]"),
    (r"\b\d{1,2} (?:de )?(?:%s) (?:de )?\d{4}\b" % MONTHS, "[DOB]"),
    (r"\b(?:%s) \d{1,2},? \d{4}\b" % MONTHS, "[DOB]"),
    (r"\b\d-\d{4}-\d{4}\b", "[ID]"),
    (r"\+?\d{3}[ -]?\d{4}[ -]?\d{4}\b", "[PHONE]"),
    (r"\b\d{4}[ -]?\d{4}\b", "[PHONE]"),
    (r"\b\d{9}\b", "[ID]"),
]

redacted = {}
for doc in docs:
    text = doc["text"]

    # hide names found after a greeting or sign-off, everywhere in the text
    names = re.findall(r"(?:Hi|Hello|Dear|Hola|Regards|Saludos|Thanks|Gracias),?\s+([A-Z]\w+(?: [A-Z]\w+)?)", text)
    # new: the part of the email before the @ is personal too
    names += re.findall(r"([\w.+-]+)@", text)
    for name in names:
        for word in name.split():
            text = text.replace(word, "[NAME]")

    for pattern, tag in PATTERNS:
        text = re.sub(pattern, tag, text)

    # new: a capitalized word in the middle of a sentence is probably a name or a place
    text = re.sub(r"(?<=[a-záéíóúñ,\d] )[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", "[NAME]", text)

    redacted[doc["id"]] = text

json.dump({"redacted_documents": redacted}, open("test-output.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)