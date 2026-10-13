"""Run the full mini-RAG pipeline and write every artifact.

    python run_pipeline.py                       # extractive (no LLM), defaults
    python run_pipeline.py --kb other_kb --queries other.json --out out_dir
    python run_pipeline.py --llm gemini          # optional; needs GEMINI_API_KEY

Exit code 0 only if every stage ran and all artifacts validated.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from rag import answering, chunking, config, evaluation, loader, retrieval, validation, vocab
from rag.jsonio import ArtifactStore
from rag.stages import Pipeline, Stage, StageOrderError, stage_step, utc_now


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", type=Path, default=config.DEFAULT_KB_DIR, help="knowledge base directory")
    p.add_argument("--queries", type=Path, default=config.DEFAULT_QUERIES_PATH, help="queries.json path")
    p.add_argument("--out", type=Path, default=config.DEFAULT_ARTIFACTS_DIR, help="artifacts directory")
    p.add_argument("--chunking", choices=vocab.CHUNKING_STRATEGIES, default=config.PRIMARY_CHUNKING)
    p.add_argument("--top-k", type=int, default=config.TOP_K,
                   help=f"chunks retrieved per query (>= {config.MIN_RETRIEVED})")
    p.add_argument("--llm", choices=("none",) + config.LLM_PROVIDERS, default="none",
                   help="optional LLM answer generator (default: none = deterministic extractive)")
    p.add_argument("--llm-model", default=None, help="LLM model id (default: $GEMINI_MODEL or config)")
    p.add_argument("--no-comparison", action="store_true", help="skip the chunking comparison")
    args = p.parse_args(argv)
    if args.top_k < config.MIN_RETRIEVED:
        p.error(f"--top-k must be >= {config.MIN_RETRIEVED}")
    return args


def build_manifest(pipe: Pipeline, args, store: ArtifactStore, corpus, answer_set,
                   status: str, warnings: list[str], error: str | None = None) -> dict:
    mode = answer_set.generator_mode if answer_set else (
        vocab.GENERATOR_EXTRACTIVE if args.llm == "none" else vocab.GENERATOR_LLM)
    manifest = {
        "pipeline_version": config.PIPELINE_VERSION,
        "status": status,
        "final_state": pipe.state.name,
        "generated_at": utc_now(),
        "stage_log": pipe.log,
        "inputs": {
            "kb_dir": loader.display_path(args.kb),
            "num_documents": len(corpus.documents) if corpus else None,
            "num_kb_files": corpus.num_files if corpus else None,
            "queries_path": loader.display_path(args.queries),
            "num_queries": len(corpus.queries) if corpus else None,
            "kb_fingerprint": corpus.fingerprint if corpus else None,
        },
        "config": config.snapshot(args.chunking, args.top_k),
        "answer_generator": mode,
        "llm": ({"provider": args.llm, "model": answer_set.llm_calls[0]["model"]
                 if answer_set and answer_set.llm_calls else args.llm_model}
                if args.llm != "none" else None),
        "artifacts_written": list(store.written) + [config.MANIFEST_FILE],
        "warnings": warnings,
    }
    if error:
        manifest["error"] = error
    return manifest


@stage_step(requires=Stage.VALIDATION_COMPLETE, produces=Stage.RESULTS_FINALISED)
def finalise_stage(pipeline) -> None:
    """Gate only: reachable solely from VALIDATION_COMPLETE. The manifest is
    written right after, so it records RESULTS_FINALISED as the final state."""


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    config.check_config()

    pipe = Pipeline()
    store = ArtifactStore(args.out)
    removed = store.clean()
    if removed:
        print(f"[INIT] removed stale artifacts: {', '.join(removed)}")

    corpus = answer_set = None
    warnings: list[str] = []
    client = None
    try:
        if args.llm != "none":
            from rag import llm
            client = llm.make_client(args.llm, args.llm_model)
            print(f"[INIT] LLM mode: {client.provider} / {client.model}")

        corpus = loader.load_stage(pipe, args.kb, args.queries)
        warnings.extend(corpus.warnings)
        print(f"[{pipe.state.name}] {len(corpus.documents)} documents, {len(corpus.queries)} queries")

        chunkset = chunking.chunk_stage(pipe, corpus, args.chunking, store)
        print(f"[{pipe.state.name}] {len(chunkset.chunks)} chunks ({chunkset.strategy})")

        index = retrieval.build_index_stage(pipe, chunkset)
        print(f"[{pipe.state.name}] {index.n} chunks indexed")

        results = retrieval.retrieve_stage(pipe, index, corpus, args.top_k, store)
        print(f"[{pipe.state.name}] top-{args.top_k} retrieved for {len(results.per_query)} queries")

        answer_set = answering.answer_stage(pipe, results, store, client)
        warnings.extend(answer_set.warnings)
        labels = [a["answer_label"] for a in answer_set.answers]
        print(f"[{pipe.state.name}] " + ", ".join(f"{lab}={labels.count(lab)}"
                                                    for lab in vocab.ANSWER_LABELS))

        evals = evaluation.evaluate_stage(pipe, results, corpus, chunkset, store,
                                          run_comparison=not args.no_comparison)
        s = evals.eval_doc["summary"]
        print(f"[{pipe.state.name}] top3_hit_rate={s['top3_hit_rate']:.3f} hits={s['hits']} "
              f"partial={s['partial_hits']} misses={s['misses']} mrr={s['mrr']:.3f}")

        run_context = {"answer_generator": answer_set.generator_mode,
                       "kb_fingerprint": corpus.fingerprint}
        outcome = validation.validation_stage(pipe, results, answer_set, store, args.kb,
                                              args.queries, run_context)
        g = outcome.grounding["summary"]
        print(f"[{pipe.state.name}] all {len(outcome.report.results)} validation rules passed; "
              f"grounding passed={g['passed']} failed={g['failed']} n/a={g['not_applicable']}")

        finalise_stage(pipe)
        store.write_json(config.MANIFEST_FILE,
                         build_manifest(pipe, args, store, corpus, answer_set, "ok", warnings))
        print(f"[{pipe.state.name}] artifacts written to {loader.display_path(args.out)}")
        for w in warnings:
            print(f"  warning: {w}")
        return 0

    except Exception as exc:  # noqa: BLE001 - every failure must end in a failed manifest
        if isinstance(exc, validation.ValidationFailed):
            print(exc.report.render(), file=sys.stderr)
        elif isinstance(exc, (loader.InputError, StageOrderError)) or type(exc).__name__ == "LLMError":
            print(f"error: {exc}", file=sys.stderr)
        else:
            traceback.print_exc()
        try:
            store.write_json(config.MANIFEST_FILE, build_manifest(
                pipe, args, store, corpus, answer_set, "failed", warnings,
                error=f"{type(exc).__name__}: {exc}"))
        except Exception as manifest_exc:  # noqa: BLE001
            print(f"could not write failure manifest: {manifest_exc}", file=sys.stderr)
        print(f"pipeline FAILED at stage {pipe.state.name}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
