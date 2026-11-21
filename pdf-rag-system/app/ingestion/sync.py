"""Decide what an ingestion run must do by comparing disk with the database.

This module is pure (no I/O) so the idempotency rules are easy to test.

Matching rules, in order:

1. Files are matched to documents by content hash first. Same hash and same
   filename -> unchanged. Same hash, different filename -> the file was
   renamed or moved; only the filename is updated (no re-embedding).
2. Files whose hash is unknown are matched by filename. A filename match means
   the file was modified -> its chunks are replaced. Otherwise it is new.
3. Documents matched by neither have no file on disk. They are KEPT by
   default, so PDFs can be removed from the folder to free space while their
   index stays searchable. They are deleted only when ``prune`` is set. A
   document whose file still exists but failed validation is always kept, so
   a transient read error never wipes an index.
4. When several files on disk have identical bytes, the first path (sorted)
   is indexed and the rest are reported as duplicates.
5. An unchanged document is reprocessed when the embedding model or chunking
   settings it was built with differ from the current configuration, or when
   ``force`` is set.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from app.ingestion.models import PDFFile
from app.vectorstore.models import DocumentRecord


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    def matches(self, doc: DocumentRecord) -> bool:
        return (
            doc.embedding_model == self.embedding_model
            and doc.chunk_size == self.chunk_size
            and doc.chunk_overlap == self.chunk_overlap
        )


@dataclass(frozen=True, slots=True)
class ProcessTask:
    """A file whose chunks must be (re)built and embedded."""

    file: PDFFile
    existing_document_id: UUID | None   # None -> insert a new document
    reason: str                         # "new" | "modified" | "config_changed" | "forced"


@dataclass(slots=True)
class SyncPlan:
    unchanged: list[PDFFile] = field(default_factory=list)
    renames: list[tuple[UUID, str, str]] = field(default_factory=list)  # (doc id, old name, new name)
    to_process: list[ProcessTask] = field(default_factory=list)
    deletions: list[DocumentRecord] = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)            # duplicate path -> indexed path
    kept_invalid: list[str] = field(default_factory=list)
    kept_missing: list[str] = field(default_factory=list)               # indexed, file no longer on disk

    @property
    def has_changes(self) -> bool:
        return bool(self.renames or self.to_process or self.deletions)

    def summary(self) -> dict[str, int]:
        reasons = [t.reason for t in self.to_process]
        return {
            "unchanged": len(self.unchanged),
            "new": reasons.count("new"),
            "modified": reasons.count("modified"),
            "config_changed": reasons.count("config_changed"),
            "forced": reasons.count("forced"),
            "renamed": len(self.renames),
            "deleted": len(self.deletions),
            "duplicates": len(self.duplicates),
            "kept_invalid": len(self.kept_invalid),
            "kept_missing": len(self.kept_missing),
        }


def build_sync_plan(
    files: Sequence[PDFFile],
    documents: Sequence[DocumentRecord],
    config: IngestionConfig,
    *,
    invalid_filenames: Collection[str] = (),
    force: bool = False,
    prune: bool = False,
) -> SyncPlan:
    plan = SyncPlan()

    # Rule 4: one file per distinct content hash.
    unique_files: list[PDFFile] = []
    first_path_for_hash: dict[str, str] = {}
    for f in sorted(files, key=lambda f: f.filename):
        if f.sha256 in first_path_for_hash:
            plan.duplicates[f.filename] = first_path_for_hash[f.sha256]
        else:
            first_path_for_hash[f.sha256] = f.filename
            unique_files.append(f)

    docs_by_hash = {d.file_hash: d for d in documents}
    docs_by_name = {d.filename: d for d in documents}
    matched: set[UUID] = set()
    unmatched_files: list[PDFFile] = []

    def needs_rebuild(doc: DocumentRecord) -> str | None:
        if force:
            return "forced"
        if not config.matches(doc):
            return "config_changed"
        return None

    # Rule 1: match by content.
    for f in unique_files:
        doc = docs_by_hash.get(f.sha256)
        if doc is None:
            unmatched_files.append(f)
            continue
        matched.add(doc.id)
        if doc.filename != f.filename:
            plan.renames.append((doc.id, doc.filename, f.filename))
        reason = needs_rebuild(doc)
        if reason:
            plan.to_process.append(ProcessTask(f, doc.id, reason))
        elif doc.filename == f.filename:
            plan.unchanged.append(f)

    # Rule 2: match the rest by filename.
    for f in unmatched_files:
        doc = docs_by_name.get(f.filename)
        if doc is not None and doc.id not in matched:
            matched.add(doc.id)
            plan.to_process.append(ProcessTask(f, doc.id, "modified"))
        else:
            plan.to_process.append(ProcessTask(f, None, "new"))

    # Rule 3: whatever is left in the database has no file on disk.
    invalid = set(invalid_filenames)
    for doc in documents:
        if doc.id in matched:
            continue
        if doc.filename in invalid:
            plan.kept_invalid.append(doc.filename)
        elif prune:
            plan.deletions.append(doc)
        else:
            plan.kept_missing.append(doc.filename)

    return plan
