import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.models import PDFFile
from app.ingestion.sync import IngestionConfig, build_sync_plan
from app.vectorstore.models import DocumentRecord

CONFIG = IngestionConfig(embedding_model="text-embedding-3-small", chunk_size=800, chunk_overlap=150)


def h(char: str) -> str:
    return char * 64


def file(name: str, digest: str) -> PDFFile:
    return PDFFile(path=Path("/data") / name, filename=name, size=100, sha256=digest)


def doc(name: str, digest: str, **kw: object) -> DocumentRecord:
    now = datetime.now(timezone.utc)
    values = dict(
        id=uuid.uuid4(), filename=name, file_hash=digest, file_size=100, total_pages=1, total_chunks=1,
        embedding_model=CONFIG.embedding_model, chunk_size=CONFIG.chunk_size, chunk_overlap=CONFIG.chunk_overlap,
        metadata={}, created_at=now, updated_at=now,
    )
    values.update(kw)
    return DocumentRecord(**values)  # type: ignore[arg-type]


def test_new_files_are_processed() -> None:
    plan = build_sync_plan([file("a.pdf", h("a")), file("b.pdf", h("b"))], [], CONFIG)
    assert [(t.file.filename, t.reason, t.existing_document_id) for t in plan.to_process] == [
        ("a.pdf", "new", None),
        ("b.pdf", "new", None),
    ]
    assert not plan.deletions and not plan.renames


def test_unchanged_files_are_skipped() -> None:
    plan = build_sync_plan([file("a.pdf", h("a"))], [doc("a.pdf", h("a"))], CONFIG)
    assert [f.filename for f in plan.unchanged] == ["a.pdf"]
    assert not plan.has_changes


def test_modified_file_replaces_existing_document() -> None:
    existing = doc("a.pdf", h("a"))
    plan = build_sync_plan([file("a.pdf", h("z"))], [existing], CONFIG)
    assert len(plan.to_process) == 1
    task = plan.to_process[0]
    assert (task.reason, task.existing_document_id) == ("modified", existing.id)
    assert not plan.deletions


def test_removed_file_keeps_its_index_by_default() -> None:
    gone = doc("gone.pdf", h("g"))
    plan = build_sync_plan([file("a.pdf", h("a"))], [doc("a.pdf", h("a")), gone], CONFIG)
    assert plan.kept_missing == ["gone.pdf"]
    assert plan.deletions == [] and not plan.has_changes


def test_removed_file_is_deleted_with_prune() -> None:
    gone = doc("gone.pdf", h("g"))
    plan = build_sync_plan([file("a.pdf", h("a"))], [doc("a.pdf", h("a")), gone], CONFIG, prune=True)
    assert plan.deletions == [gone]
    assert plan.kept_missing == []


def test_empty_folder_keeps_everything_by_default() -> None:
    docs = [doc("a.pdf", h("a")), doc("b.pdf", h("b"))]
    plan = build_sync_plan([], docs, CONFIG)
    assert plan.deletions == [] and sorted(plan.kept_missing) == ["a.pdf", "b.pdf"]


def test_readding_a_removed_file_needs_no_reprocessing() -> None:
    existing = doc("a.pdf", h("a"))
    plan = build_sync_plan([file("a.pdf", h("a"))], [existing], CONFIG)
    assert [f.filename for f in plan.unchanged] == ["a.pdf"] and not plan.to_process


def test_new_file_at_the_name_of_a_removed_file_replaces_it() -> None:
    existing = doc("a.pdf", h("a"))
    plan = build_sync_plan([file("a.pdf", h("z"))], [existing], CONFIG)
    assert [(t.reason, t.existing_document_id) for t in plan.to_process] == [("modified", existing.id)]


def test_renamed_file_is_renamed_without_reprocessing() -> None:
    existing = doc("old.pdf", h("a"))
    plan = build_sync_plan([file("new.pdf", h("a"))], [existing], CONFIG)
    assert plan.renames == [(existing.id, "old.pdf", "new.pdf")]
    assert not plan.to_process and not plan.deletions


def test_swapped_files_are_two_renames() -> None:
    a, b = doc("a.pdf", h("a")), doc("b.pdf", h("b"))
    plan = build_sync_plan([file("a.pdf", h("b")), file("b.pdf", h("a"))], [a, b], CONFIG)
    assert sorted(plan.renames) == sorted([(b.id, "b.pdf", "a.pdf"), (a.id, "a.pdf", "b.pdf")])
    assert not plan.to_process and not plan.deletions


def test_identical_files_on_disk_are_indexed_once() -> None:
    plan = build_sync_plan([file("copy.pdf", h("a")), file("a.pdf", h("a"))], [], CONFIG)
    assert [t.file.filename for t in plan.to_process] == ["a.pdf"]
    assert plan.duplicates == {"copy.pdf": "a.pdf"}


def test_config_change_triggers_reprocessing() -> None:
    existing = doc("a.pdf", h("a"), chunk_size=500)
    plan = build_sync_plan([file("a.pdf", h("a"))], [existing], CONFIG)
    assert [(t.reason, t.existing_document_id) for t in plan.to_process] == [("config_changed", existing.id)]


def test_force_reprocesses_everything() -> None:
    existing = doc("a.pdf", h("a"))
    plan = build_sync_plan([file("a.pdf", h("a"))], [existing], CONFIG, force=True)
    assert [t.reason for t in plan.to_process] == ["forced"]


def test_unreadable_file_keeps_its_index() -> None:
    existing = doc("broken.pdf", h("a"))
    plan = build_sync_plan([], [existing], CONFIG, invalid_filenames={"broken.pdf"})
    assert plan.kept_invalid == ["broken.pdf"]
    assert not plan.deletions


def test_new_content_at_path_whose_old_content_moved() -> None:
    # a.pdf was renamed to b.pdf, and a new file was created at a.pdf.
    existing = doc("a.pdf", h("a"))
    plan = build_sync_plan([file("a.pdf", h("n")), file("b.pdf", h("a"))], [existing], CONFIG)
    assert plan.renames == [(existing.id, "a.pdf", "b.pdf")]
    assert [(t.file.filename, t.reason) for t in plan.to_process] == [("a.pdf", "new")]
