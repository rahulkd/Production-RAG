1. Your e2e evaluation calls /ask and /stream. A rerun within 6 hours (ttl_hours) gets cached answers back almost instantly, so the latency numbers are meaningless and the generation step isn't actually tested. 

2. Before an eval run, either flush Redis (docker exec rag-redis redis-cli FLUSHALL) or set REDIS__TTL_HOURS very low.