from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import pandas as pd

def load_metric(path: Path):
    spec=importlib.util.spec_from_file_location('official_kaggle_metric', path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def build_solution(gold_csv: Path) -> pd.DataFrame:
    g=pd.read_csv(gold_csv)
    rows=[]
    for doc,d in g.groupby('documento_id', sort=False):
        parts=[]
        for _,r in d.sort_values('inicio').iterrows():
            idc='-' if pd.isna(r.id_canonico) else str(int(r.id_canonico))
            parts.append(f"{int(r.inicio)},{int(r.fim)},{r.classificacao},{idc}")
        rows.append({'documento_id':doc,'nivel':int(d.nivel.iloc[0]),'citacoes':'|'.join(parts)})
    return pd.DataFrame(rows)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--submission', type=Path, required=True)
    ap.add_argument('--report', type=Path, default=Path('reports/current_errors.json'))
    args=ap.parse_args()

    # Snapshot oficial (data/current/) ou layout do repositório (data/).
    metric_path=args.data_dir/'kaggle_metric.py'
    gold_path=args.data_dir/'goldenset_offsets.csv'
    metric=load_metric(metric_path if metric_path.exists() else Path('kaggle_metric.py'))
    sol=build_solution(gold_path if gold_path.exists() else args.data_dir/'goldenset.csv')
    sub=pd.read_csv(args.submission)
    res=metric.avaliar(sol,sub)
    print(json.dumps(res,ensure_ascii=False,indent=2))

    errors=[]
    sol_idx=sol.set_index('documento_id')
    sub_idx=sub.set_index('documento_id')
    for doc,line in sol_idx.iterrows():
        golds=metric._parse_solution_cell(line['citacoes'],doc)
        preds=metric._parse_submission_cell(sub_idx.loc[doc,'citacoes'],doc) if doc in sub_idx.index else []
        pares,g_sem,p_sem=metric._casar(golds,preds)
        matched_golds=[golds[gi] for gi,_ in pares]
        with open(args.data_dir/'txt'/f'{doc}.txt',encoding='utf-8',newline='') as f: text=f.read()
        for gi,pi in pares:
            g,p=golds[gi],preds[pi]
            link_ok=(g['classe']!='real' or p['id_canonico'] in g['doc_ids'])
            if g['classe']!=p['classe'] or not link_ok:
                errors.append({'documento_id':doc,'kind':'matched_wrong','trecho':text[g['inicio']:g['fim']], 'gold':g,'pred':p})
        for gi in g_sem:
            g=golds[gi]
            errors.append({'documento_id':doc,'kind':'missing','trecho':text[g['inicio']:g['fim']], 'gold':g,'pred':None})
        for pi in p_sem:
            p=preds[pi]
            if not any(metric._contida(p,g) for g in matched_golds):
                errors.append({'documento_id':doc,'kind':'extra','trecho':text[p['inicio']:p['fim']], 'gold':None,'pred':p})

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(errors,ensure_ascii=False,indent=2,default=lambda x:list(x) if isinstance(x,(set,frozenset)) else str(x)),encoding='utf-8')
    counts={}
    for e in errors: counts[e['kind']]=counts.get(e['kind'],0)+1
    print(f"\nerrors: {len(errors)} {counts}")
    print(f"report: {args.report}")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
