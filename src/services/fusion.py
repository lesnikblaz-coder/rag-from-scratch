def reciprocal_rank_fusion(
        semantic_results: list[dict],
        keyword_results: list[dict],
        k: int = 60
) -> list[dict]:

    # store each document's accumulated RRF score
    fused = {}

    for results in (semantic_results, keyword_results):
        for rank, document in enumerate(results, start=1):
            key = (document["source_file"], document["chunk_index"])

            if key not in fused:
                fused[key] = {**document, "rrf_score": 0}

            fused[key]["rrf_score"] += 1 / (k + rank)


    return sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)