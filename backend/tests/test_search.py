from unittest.mock import Mock
from uuid import UUID

import pytest

from app import search
from app.search import Bm25Hit, VectorHit, fuse_rrf

DOCUMENT_ID = UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276")


def test_rrf_combines_positions_from_both_rankings():
    bm25_hits = [
        Bm25Hit(1, DOCUMENT_ID, ["Заявка"], "Чанк A", 5.0),
        Bm25Hit(2, DOCUMENT_ID, ["Заявка", "Сроки"], "Чанк B", 1.0),
    ]
    vector_hits = [
        VectorHit(3, DOCUMENT_ID, ["Заявка"], "Чанк C", 0.1),
        VectorHit(2, DOCUMENT_ID, ["Заявка", "Сроки"], "Чанк B", 0.4),
    ]

    results = fuse_rrf(bm25_hits, vector_hits, limit=3)

    assert [hit.chunk_id for hit in results] == [2, 1, 3]
    assert [hit.rrf_score for hit in results] == pytest.approx([2 / 62, 1 / 61, 1 / 61])
    assert results[0].document_id == DOCUMENT_ID
    assert results[0].heading_path == ["Заявка", "Сроки"]
    assert results[0].text == "Чанк B"
    assert fuse_rrf(bm25_hits, vector_hits, limit=1) == results[:1]


def test_rrf_keeps_vector_ranking_when_bm25_returns_no_hits():
    vector_hits = [
        VectorHit(3, DOCUMENT_ID, ["Заявка"], "Чанк C", 0.1),
        VectorHit(1, DOCUMENT_ID, ["Заявка"], "Чанк A", 0.4),
    ]

    results = fuse_rrf([], vector_hits, limit=5)

    assert [hit.chunk_id for hit in results] == [3, 1]
    assert [hit.rrf_score for hit in results] == pytest.approx([1 / 61, 1 / 62])


def test_rrf_returns_no_hits_for_empty_input_or_zero_limit():
    assert fuse_rrf([], [], limit=5) == []
    assert fuse_rrf([Bm25Hit(1, DOCUMENT_ID, ["Заявка"], "Чанк A", 5.0)], [], limit=0) == []


@pytest.mark.parametrize(("limit", "candidate_limit", "ids"), [(1, 30, [2]), (40, 40, [2, 1, 3])])
def test_hybrid_search_limits_results_after_fusion(monkeypatch, limit, candidate_limit, ids):
    session = Mock()
    bm25 = Mock(
        return_value=[
            Bm25Hit(1, DOCUMENT_ID, ["Заявка"], "Чанк A", 5.0),
            Bm25Hit(2, DOCUMENT_ID, ["Заявка"], "Чанк B", 1.0),
        ]
    )
    vector = Mock(
        return_value=[
            VectorHit(3, DOCUMENT_ID, ["Заявка"], "Чанк C", 0.1),
            VectorHit(2, DOCUMENT_ID, ["Заявка"], "Чанк B", 0.4),
        ]
    )
    monkeypatch.setattr(search, "search_bm25", bm25)
    monkeypatch.setattr(search, "search_vector", vector)

    results = search.search_hybrid(session, " заявка ", limit)

    assert [hit.chunk_id for hit in results] == ids
    bm25.assert_called_once_with(session, "заявка", candidate_limit)
    vector.assert_called_once_with(session, "заявка", candidate_limit)
