import json
import pandas as pd

df = pd.read_csv("test-input.csv")

# The numbers in an address (street, apartment, zip) never change between
# copies of a record, only their order does. So I sort the digits and add the
# zip code, and use that as the ID of the person.
def make_key(address):
    digits = "".join(sorted(c for c in address if c.isdigit()))
    zip_code = address.split()[-1]
    return zip_code + digits

df["key"] = df["address"].apply(make_key)

matches = []
for key, group in df.groupby("key"):
    ids = list(group["record_id"])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            matches.append([ids[i], ids[j]])

json.dump({"matches": matches}, open("test-output.json", "w"))
print(len(matches), "pairs")