"""Write need texts for test/validation tasks and synthetic projector-training tasks.

    python scripts/make_needs.py --datasets games ind ml --synth 8000
"""
import argparse
import pickle

import _path  # noqa: F401
from recteller import config as C, data, needs
from recteller.llm import LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--synth", type=int, default=8000)
    args = ap.parse_args()
    m = LLM()
    for ds in args.datasets:
        d = data.load(ds)
        todo = [t for t in d.test + d.val if not t.need]
        if todo:
            for t, s in zip(todo, needs.write_needs(m, d.cat, [t.need_attrs for t in todo])):
                t.need = s
            data.save(d)
        sp = C.PROC / f"{ds}_synth.pkl"
        if not sp.exists() and args.synth:
            excl = {t.target for t in d.test + d.val}
            syn = data.make_synth_tasks(d.cat, args.synth, excl)
            for t, s in zip(syn, needs.write_needs(m, d.cat, [t.need_attrs for t in syn])):
                t.need = s
            pickle.dump(syn, open(sp, "wb"))
        print(ds, "done", d.test[0].need, flush=True)


if __name__ == "__main__":
    main()
