from core.vector_store import VectorStore
from core.search_router import SearchRouter

vs = VectorStore()
vs.initialize()

router = SearchRouter(vector_store=vs)
result = router.search("Python developer", top_k=3, search_mode="shallow")

for r in result.get("results", []):
    print(f"Name: {r.get('name')}")
    print(f"Metadata keys: {list(r.get('metadata', {}).keys())}")
    print(f"Metadata type field: {r.get('metadata', {}).get('type', 'MISSING')}")
    print(f"Metadata name field: {r.get('metadata', {}).get('name', 'MISSING')}")
    print(f"page_content preview: {r.get('page_content', '')[:150]}")
    print("---")
