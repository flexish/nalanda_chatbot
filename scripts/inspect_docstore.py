import pickle
import sys
import types
from pathlib import Path

# ensure project root is on sys.path so pickled classes in repo can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

p = project_root / "vectorstore_test" / "docstore.pkl"
# stub missing heavy deps that some repo modules import at top-level during pickle import
if 'unstructured_pytesseract' not in sys.modules:
    m = types.ModuleType('unstructured_pytesseract')
    # provide minimal pytesseract attr used by repo import code
    m.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    sys.modules['unstructured_pytesseract'] = m
print("docstore path:", p)
if not p.exists():
    print("docstore not found")
    raise SystemExit(1)

ds = pickle.load(open(p, "rb"))
print("loaded", type(ds), "len=", len(ds))
count_image = 0
for i, (k, v) in enumerate(ds.items()):
    kind = getattr(v, 'kind', None)
    metadata_kind = None
    try:
        metadata_kind = v.metadata.get('kind') if getattr(v, 'metadata', None) else None
    except Exception:
        metadata_kind = None
    text_len = len(getattr(v, 'text', '') or '')
    print(i, 'id=', k, 'kind=', kind, 'metadata_kind=', metadata_kind, 'text_len=', text_len)
    if (kind == 'image') or (metadata_kind == 'image'):
        count_image += 1
print('image parents:', count_image)
