# List all indexes in OpenSearch
from opensearchpy import OpenSearch

HOST = "http://localhost:9200"

client = OpenSearch(
    hosts=[HOST],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

print(f"OpenSearch indexes @ {HOST}")
print("=" * 60)

# cat.indices returns one dict per index. include_system=False hides
# internal indices (those starting with a dot, e.g. .opendistro-*).
indices = client.cat.indices(format="json", bytes="b")

# Keep user indices (drop dot-prefixed system indices), sorted by name.
user_indices = sorted(
    (idx for idx in indices if not idx["index"].startswith(".")),
    key=lambda i: i["index"],
)

if not user_indices:
    print("No user indexes found.")
else:
    print(f"{'index':<30} {'health':<7} {'docs':>8} {'size':>12}")
    print("-" * 60)
    for idx in user_indices:
        name = idx["index"]
        health = idx.get("health", "?")
        docs = idx.get("docs.count") or "0"
        size_bytes = int(idx.get("store.size") or 0)
        print(f"{name:<30} {health:<7} {int(docs):>8} {size_bytes:>10,} B")

    print("-" * 60)
    print(f"Total user indexes: {len(user_indices)}")
