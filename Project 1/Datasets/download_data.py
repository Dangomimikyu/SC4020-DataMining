import gzip
import hashlib
import os
import shutil
import urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

SNAP = "https://snap.stanford.edu/data"
PLANETOID = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"

FILES = [
    (f"{SNAP}/email-Eu-core.txt.gz", "email-Eu-core.txt", True),
    (f"{SNAP}/email-Eu-core-department-labels.txt.gz",
     "email-Eu-core-department-labels.txt", True),
] + [
    (f"{PLANETOID}/ind.cora.{k}", f"ind.cora.{k}", False)
    for k in ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
]

EXPECTED_LINES = {
    "email-Eu-core.txt": 25571,
    "email-Eu-core-department-labels.txt": 1005,   
}


def fetch(url, name, gz):
    dest = os.path.join(RAW, name)
    if os.path.exists(dest):
        print(f"  have  {name}")
        return dest

    print(f"  get   {name}")
    if gz:
        tmp = dest + ".gz"
        urllib.request.urlretrieve(url, tmp)
        with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(tmp)
    else:
        urllib.request.urlretrieve(url, dest)
    return dest


def main():
    os.makedirs(RAW, exist_ok=True)
    print(f"downloading into {os.path.abspath(RAW)}")

    for url, name, gz in FILES:
        path = fetch(url, name, gz)

        if name in EXPECTED_LINES:
            with open(path) as f:
                n = sum(1 for _ in f)
            want = EXPECTED_LINES[name]
            flag = "ok" if n == want else f"MISMATCH (expected {want})"
            print(f"        {n} lines -- {flag}")

        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:12]
        print(f"        sha256:{digest}")

    print("\ndone. now run: python graph_datasets.py")


if __name__ == "__main__":
    main()
