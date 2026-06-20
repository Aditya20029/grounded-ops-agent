"""Evaluation harness: runs the gold set and computes the metrics report.

Retrieval metrics (recall@k, nDCG@k, MRR) are doc-level. Citation metrics use the
sources the generator actually cited. Faithfulness (LLM judge, temperature 0) is
optional and only run when a judge model is supplied. A FAISS-vs-pgvector
comparison reports recall@k and p50/p95 latency on the gold queries. Numbers are
reproducible given the seeded data and pinned models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from app.agent.faithfulness import score_faithfulness
from app.agent.generation import generate_grounded_answer
from app.core.settings import Settings
from app.eval import metrics
from app.eval.gold import GoldItem, load_gold
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.retrieval.faiss_store import DEFAULT_INDEX_DIR, FaissStore
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.service import RetrievalService, build_retrieval_service
from app.retrieval.vector_store import VectorStore


@dataclass
class StoreBenchmark:
    label: str
    recall_at_k: float
    p50_ms: float
    p95_ms: float


@dataclass
class EvalReport:
    n: int
    top_k: int
    models: dict[str, str]
    date: str
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    citation_precision: float
    citation_recall: float
    faithfulness: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    avg_cost_usd: float
    benchmark: list[StoreBenchmark] = field(default_factory=list)


async def _doc_ranking(
    store: VectorStore, provider_query_vec: list[float], top_k: int
) -> list[str]:
    chunks = await store.search(provider_query_vec, top_k * 3)
    return metrics.dedup_preserve_order([c.doc_id for c in chunks])[:top_k]


async def _benchmark(
    service: RetrievalService, gold: list[GoldItem], top_k: int, dim: int, model: str
) -> list[StoreBenchmark]:
    pg = service.pg_store
    faiss = FaissStore(dim, index_path=DEFAULT_INDEX_DIR / "eval_flat.faiss", index_type="flat")
    await faiss.ensure(pg, model)

    rows: list[StoreBenchmark] = []
    for label, store in (("pgvector", pg), ("faiss:flat", faiss)):
        recalls: list[float] = []
        latencies: list[float] = []
        for item in gold:
            qv = await _embed(service, item.question)
            start = perf_counter()
            ranking = await _doc_ranking(store, qv, top_k)
            latencies.append((perf_counter() - start) * 1000.0)
            recalls.append(metrics.recall_at_k(ranking, set(item.relevant_doc_ids), top_k))
        rows.append(
            StoreBenchmark(
                label=label,
                recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
                p50_ms=metrics.percentile(latencies, 0.50),
                p95_ms=metrics.percentile(latencies, 0.95),
            )
        )
    return rows


async def _embed(service: RetrievalService, text: str) -> list[float]:
    import asyncio

    return await asyncio.to_thread(service.provider.embed_query, text)


async def run_eval(
    settings: Settings,
    *,
    top_k: int = 5,
    date: str,
    judge: LLMProvider | None = None,
) -> EvalReport:
    gold = load_gold()
    service = build_retrieval_service(settings)
    llm = get_llm_provider(settings)

    recalls: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []
    cite_p: list[float] = []
    cite_r: list[float] = []
    faith: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []

    for item in gold:
        relevant = set(item.relevant_doc_ids)
        qv = await _embed(service, item.question)
        start = perf_counter()
        chunks = await PgVectorStore().search(qv, top_k * 3)
        latencies.append((perf_counter() - start) * 1000.0)
        ranking = metrics.dedup_preserve_order([c.doc_id for c in chunks])[:top_k]
        recalls.append(metrics.recall_at_k(ranking, relevant, top_k))
        ndcgs.append(metrics.ndcg_at_k(ranking, relevant, top_k))
        rrs.append(metrics.mrr(ranking, relevant))

        answer = await generate_grounded_answer(item.question, service, llm, top_k=top_k)
        cited = {s.doc_id for s in answer.sources}
        precision, recall = metrics.citation_precision_recall(cited, relevant)
        cite_p.append(precision)
        cite_r.append(recall)
        costs.append(answer.cost_usd)

        if judge is not None and answer.sources:
            result = await score_faithfulness(
                answer.answer, [s.snippet for s in answer.sources], judge
            )
            faith.append(result.score)

    n = len(gold)
    bench = await _benchmark(service, gold, top_k, settings.embedding_dim, llm.model_name)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return EvalReport(
        n=n,
        top_k=top_k,
        models={
            "llm": settings.llm_model if settings.llm_provider != "fake" else "echo-offline",
            "embedding": service.provider.model_name,
            "judge": (settings.eval_judge_model if judge is not None else "none"),
        },
        date=date,
        recall_at_k=mean(recalls),
        ndcg_at_k=mean(ndcgs),
        mrr=mean(rrs),
        citation_precision=mean(cite_p),
        citation_recall=mean(cite_r),
        faithfulness=mean(faith) if faith else None,
        latency_p50_ms=metrics.percentile(latencies, 0.50),
        latency_p95_ms=metrics.percentile(latencies, 0.95),
        avg_cost_usd=mean(costs),
        benchmark=bench,
    )


def format_report(report: EvalReport) -> str:
    faith = f"{report.faithfulness:.3f}" if report.faithfulness is not None else "n/a (no judge)"
    lines = [
        f"## Evaluation ({report.n} gold items, top_k={report.top_k})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| recall@{report.top_k} | {report.recall_at_k:.3f} |",
        f"| nDCG@{report.top_k} | {report.ndcg_at_k:.3f} |",
        f"| MRR | {report.mrr:.3f} |",
        f"| Citation precision | {report.citation_precision:.3f} |",
        f"| Citation recall | {report.citation_recall:.3f} |",
        f"| Faithfulness (3-pt) | {faith} |",
        f"| p50 latency (retrieval) | {report.latency_p50_ms:.1f} ms |",
        f"| p95 latency (retrieval) | {report.latency_p95_ms:.1f} ms |",
        f"| Avg cost / query | ${report.avg_cost_usd:.5f} |",
        "",
        f"### FAISS vs pgvector (recall@{report.top_k}, latency)",
        "",
        "| store | recall@k | p50 ms | p95 ms |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.benchmark:
        lines.append(
            f"| {row.label} | {row.recall_at_k:.3f} | {row.p50_ms:.2f} | {row.p95_ms:.2f} |"
        )
    lines += [
        "",
        f"_Models: llm={report.models['llm']}, embedding={report.models['embedding']}, "
        f"judge={report.models['judge']}. Date: {report.date}. Seeded + temperature 0._",
    ]
    return "\n".join(lines)
