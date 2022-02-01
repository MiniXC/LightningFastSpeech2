import pandas as pd
import numpy as np
from unicodedata import normalize

_df1 = pd.read_csv(
    "https://raw.githubusercontent.com"
    + "/dmort27/panphon/master/panphon/data/ipa_all.csv"
)
_df2 = pd.read_csv(
    "https://raw.githubusercontent.com"
    + "/dmort27/panphon/master/panphon/data/ipa_bases.csv"
)
_df1["ipa"] = _df1["ipa"].apply(lambda x: normalize("NFC", x))
_df2["ipa"] = _df2["ipa"].apply(lambda x: normalize("NFC", x))
ipa_df = pd.concat([_df1, _df2]).set_index("ipa")
ipa_weights_df = pd.read_csv(
    "https://raw.githubusercontent.com"
    + "/dmort27/panphon/master/panphon/data/feature_weights.csv"
)

for c in ipa_df.columns.tolist():
    if c not in ipa_weights_df.columns:
        ipa_weights_df[c] = [0.5]

ipa_weights_df = ipa_weights_df[ipa_df.columns]


def to_id(x):
    new_vals = []
    for k in x:
        if k == "0":
            new_vals += [1]
        elif k == "+":
            new_vals += [2]
        elif k == "-":
            new_vals += [0]
    return new_vals


ipa_df = ipa_df.apply(to_id, axis=1, raw=True)

de2en_map = {
    "t": "t̪",
    "ð": "d̪",
    "ʃ": "ʃ̺",
    "w": "v",
    "θ": "s",
    "r": "ʀ",
    "n": "n̪",
    "l": "l̪",
    "d": "d̪",
}


def get_phone_vecs(phone_list):
    return np.array([phone_vec(p) for p in phone_list])


def phone_vec(phone):
    phone = normalize("NFC", phone)
    result = []
    if phone in ipa_df.index:
        vals = ipa_df.loc[phone].values
        if len(ipa_df.loc[phone].shape) > 1:
            vals = np.mean(vals, axis=0)
        result.append(vals)
    else:
        result = [phone_vec(p) for p in phone]
    return np.mean(np.array(result), axis=0).round().astype(int)
