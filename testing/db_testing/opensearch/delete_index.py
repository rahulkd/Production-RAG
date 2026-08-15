# Delete OpenSearch indexes.
#
# Configure the behavior with the variables below (no command-line args).
# DRY_RUN is True by default so nothing is deleted until you explicitly flip it.
from opensearchpy import OpenSearch

# ============================ CONFIGURATION ============================
HOST = "http://localhost:9200"

# What to delete:
#   DELETE_ALL = True   -> delete every (user) index
#   DELETE_ALL = False  -> delete only the single index named in INDEX_NAME
DELETE_ALL = True
INDEX_NAME = "arxiv-papers-chunks"

# Safety switches:
INCLUDE_SYSTEM = False   # when DELETE_ALL: also delete dot-prefixed system indices (dangerous)
DRY_RUN = False           # True = only print what WOULD be deleted; set False to actually delete
# ======================================================================

client = OpenSearch(
    hosts=[HOST],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)


def user_indices():
    """All index names, excluding dot-prefixed system indices unless INCLUDE_SYSTEM."""
    names = [idx["index"] for idx in client.cat.indices(format="json")]
    if INCLUDE_SYSTEM:
        return sorted(names)
    return sorted(n for n in names if not n.startswith("."))


def resolve_targets():
    """Return the list of index names to delete based on the configuration."""
    if DELETE_ALL:
        return user_indices()
    if not client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' does not exist — nothing to delete.")
        return []
    return [INDEX_NAME]


def main():
    mode = "ALL user indexes" if DELETE_ALL else f"index '{INDEX_NAME}'"
    print(f"Target @ {HOST}: {mode}")
    print(f"DRY_RUN = {DRY_RUN}  |  INCLUDE_SYSTEM = {INCLUDE_SYSTEM}")
    print("=" * 60)

    targets = resolve_targets()
    if not targets:
        print("No matching indexes.")
        return

    print(f"{len(targets)} index(es) selected:")
    for name in targets:
        print(f"  - {name}")
    print("-" * 60)

    if DRY_RUN:
        print("DRY RUN: nothing deleted. Set DRY_RUN = False to delete for real.")
        return

    deleted, failed = 0, 0
    for name in targets:
        try:
            client.indices.delete(index=name)
            print(f"  ✓ deleted {name}")
            deleted += 1
        except Exception as e:
            print(f"  ✗ failed {name}: {e}")
            failed += 1

    print("-" * 60)
    print(f"Done. Deleted {deleted}, failed {failed}.")


if __name__ == "__main__":
    main()
