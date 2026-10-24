"""Every parser output is grammar-valid or UNPARSED (grammar fuzz)."""
import json
import random
import string

import _path  # noqa: F401
from recteller.intents import Compare, Details, Personalize, Search, from_json


def test_valid_forms():
    assert isinstance(from_json({"intent": "SEARCH", "constraints": [["price", "<=", "40"]], "text": "x"}, 100), Search)
    assert isinstance(from_json({"intent": "COMPARE", "items": [1, 2]}, 100), Compare)
    assert isinstance(from_json({"intent": "DETAILS", "item": 3, "attributes": ["price"]}, 100), Details)
    assert isinstance(from_json({"intent": "PERSONALIZE", "scope": ["me"], "session": [4]}, 100), Personalize)


def test_rejects_out_of_grammar():
    bad = [{"intent": "ADMIN"}, {"intent": "SEARCH", "limit": 500}, {"intent": "COMPARE", "items": [0]},
           {"intent": "COMPARE", "items": [101]}, {"intent": "SEARCH", "constraints": [["a", "!=", "b"]]},
           {"intent": "PERSONALIZE", "scope": "Ben"}, "not json", [1, 2], None,
           {"intent": "COMPARE", "items": list(range(1, 80))}]
    for b in bad:
        assert from_json(b, 100) is None, b


def test_fuzz():
    rnd = random.Random(0)
    for _ in range(1000):
        s = "".join(rnd.choice(string.printable) for _ in range(rnd.randint(0, 80)))
        r = from_json(s, 100)
        assert r is None or isinstance(r, (Search, Compare, Details, Personalize))
