import os, sys, yaml, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.vector_store import VectorStore
from src.encryption.aes_encryption import AESEncryption

cfg = yaml.safe_load(open('config/config.yaml','r',encoding='utf-8'))
vs = VectorStore(collection_name=cfg['vector_db']['collection_name'], dimension=cfg['embedding']['dimension'], distance_metric=cfg['vector_db']['distance_metric'], host=cfg['vector_db']['host'], port=cfg['vector_db']['port'])
enc = AESEncryption(key_size=cfg['encryption']['key_size'])
enc.load_key('encryption.key')

out = {'total_fetched': 0, 'samples': [], 'found_targets': {}}

recs = vs.get_all_points(batch_size=10, with_payload=True, with_vectors=False)
out['total_fetched'] = len(recs)

targets = ['20260121_1000.txt','20260701_1000.txt']
for i, rec in enumerate(recs[:20]):
    payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
    pd = vs._normalize_payload(payload)
    keys = list(pd.keys())
    sample = {'id': getattr(rec,'id',None), 'keys': keys[:20]}
    src = pd.get('source_file') or pd.get('source') or pd.get('source_file_name') or ''
    sample['source_file'] = src
    # try decrypt
    ct = pd.get('ciphertext') or pd.get('ct')
    nonce = pd.get('nonce') or pd.get('n')
    if ct and nonce:
        try:
            plain = enc.decrypt(ct, nonce)
        except Exception as e:
            plain = f'<decrypt failed: {e}>'
    else:
        plain = pd.get('text') or pd.get('plaintext') or pd.get('content')
    sample['preview'] = (plain[:400] if isinstance(plain,str) else str(plain))
    out['samples'].append(sample)
    b = os.path.basename(str(src))
    if b in targets:
        out['found_targets'].setdefault(b, []).append({'id': sample['id'], 'preview': sample['preview']})

os.makedirs('logs', exist_ok=True)
with open('logs/test_get_points.json','w',encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('WROTE logs/test_get_points.json')

