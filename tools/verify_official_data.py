from __future__ import annotations
import argparse, csv, hashlib, sqlite3
from pathlib import Path

EXPECTED = {
    "desafio1_bracis.db": "78f0708b0a21c11655dfdd882382fea75c62a75415d8d3b118888c0a340bef4c",
    "goldenset_offsets.csv": "562e4ee5d0e8cb299ccb99b4c6dd758b195fbb5668617ce4c2ead465ea27211d",
    "json_to_submission.py": "c6ec4963e884c7fc19939816d7398e512cc8f7d60af472fb1a3bd723f1fee05c",
    "kaggle_metric.py": "3c4d30e70971144afbd0ae73c6d4ac887faf0f5926de986170de32f72544fc3f",
    "sample_submission.csv": "c299ddb54b94d6375de4e58ecad8fec55a68f4cced667e19b3f4f9f60af4ffdc",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--skip-hash", action="store_true", help="Valida estrutura/contagens sem exigir exatamente o snapshot de 15/09/2026.")
    args = ap.parse_args()
    d = args.data_dir
    errors=[]
    for name, expected in EXPECTED.items():
        p=d/name
        if not p.exists():
            errors.append(f"arquivo ausente: {p}")
            continue
        actual=sha256(p)
        status="OK" if actual==expected else "DIFERENTE"
        print(f"{name:24} {status}  {actual}")
        if not args.skip_hash and actual!=expected:
            errors.append(f"hash inesperado para {name}")
    txt=d/'txt'
    docs=sorted(txt.glob('*.txt')) if txt.exists() else []
    print(f"txt documents: {len(docs)}")
    if len(docs)!=26: errors.append(f"esperados 26 .txt, encontrados {len(docs)}")

    gold=d/'goldenset_offsets.csv'
    if gold.exists():
        rows=list(csv.DictReader(gold.open(encoding='utf-8-sig', newline='')))
        by={}
        for r in rows:
            key=(int(r['nivel']),r['classificacao'])
            by[key]=by.get(key,0)+1
        print(f"gold citations: {len(rows)}")
        print("gold breakdown:", dict(sorted(by.items())))
        expected_counts={(1,'real'):52,(1,'inventada'):32,(1,'incompleta'):15,
                         (2,'real'):44,(2,'inventada'):32,(2,'incompleta'):17}
        if len(rows)!=192 or by!=expected_counts:
            errors.append("contagens do gold não correspondem ao snapshot atual esperado (192 citações)")

    db=d/'desafio1_bracis.db'
    if db.exists():
        con=sqlite3.connect(db)
        try:
            total=con.execute('select count(*) from documentos').fetchone()[0]
            naturezas=dict(con.execute('select natureza,count(*) from documentos group by natureza').fetchall())
        finally:
            con.close()
        print(f"db total: {total}; naturezas: {naturezas}")
        if total!=1014 or naturezas!={'acordao':996,'dispositivo':13,'sumula':5}:
            errors.append("contagens do SQLite não correspondem ao snapshot atual esperado")

    if errors:
        print("\nFALHOU:")
        for e in errors: print(" -",e)
        return 1
    print("\nOK: snapshot oficial atual validado.")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
