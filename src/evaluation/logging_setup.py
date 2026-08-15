"""Logging for the evaluation CLIs.

Two requirements drive this module:

1. Errors must be visible in the terminal *and* captured in a log file.
2. Papers whose PDF failed to download or parse must end up in their own
   file, listed by arXiv ID, so the list can be acted on without grepping a
   full run log.

:class:`FailureRecorder` handles the second. It is deliberately a plain
collector rather than a logging handler, because a "failure" here is a
pipeline outcome (docling returned nothing) and not necessarily an
exception — the ingestion path treats parse failures as non-fatal and logs
them at WARNING.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure root logging to write to both the terminal and a file.

    The file always captures DEBUG so a failed run is diagnosable after the
    fact; the console stays at INFO (or DEBUG with ``verbose``) so progress is
    readable while it runs.

    Third-party loggers that are noisy at DEBUG (httpx, opensearch, docling,
    botocore) are pinned to WARNING so they don't bury the pipeline's own
    output.

    :param log_file: Destination for the full run log. Parent dirs are created.
    :param verbose: Emit DEBUG to the console as well as the file.
    :returns: The configured root logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Drop handlers from any previous setup_logging call in the same process,
    # otherwise repeated CLI invocations in one session duplicate every line.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "opensearch", "urllib3", "botocore", "boto3", "docling", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Logging to {log_file}")
    return root


@dataclass
class Failure:
    """One paper that did not make it cleanly through the pipeline.

    :param arxiv_id: The paper's arXiv ID.
    :param stage: Where it failed — ``fetch``, ``download``, ``parse``,
        ``empty_text``, ``chunk``, or ``index``.
    :param reason: Human-readable cause.
    :param title: Paper title when known, for readability in the report.
    """

    arxiv_id: str
    stage: str
    reason: str
    title: str = ""


@dataclass
class FailureRecorder:
    """Collects per-paper failures and writes them to a dedicated file.

    :param failures: Accumulated failures, in the order recorded.
    """

    failures: List[Failure] = field(default_factory=list)

    def record(self, arxiv_id: str, stage: str, reason: str, title: str = "") -> None:
        """Record a failure and log it at ERROR so it also hits the terminal.

        :param arxiv_id: The paper's arXiv ID.
        :param stage: Pipeline stage that failed.
        :param reason: Human-readable cause.
        :param title: Optional paper title.
        """
        self.failures.append(Failure(arxiv_id=arxiv_id, stage=stage, reason=reason, title=title))
        logging.getLogger(__name__).error(f"[{stage}] {arxiv_id}: {reason}")

    def by_stage(self) -> Dict[str, List[Failure]]:
        """Group the recorded failures by pipeline stage.

        :returns: Stage name to the failures recorded for it.
        """
        grouped: Dict[str, List[Failure]] = {}
        for failure in self.failures:
            grouped.setdefault(failure.stage, []).append(failure)
        return grouped

    def __len__(self) -> int:
        return len(self.failures)

    def write(self, text_path: Path, json_path: Optional[Path] = None) -> None:
        """Write the failure report.

        Always writes the text file, even when there are no failures — an
        empty report is meaningful evidence that the run was clean, whereas a
        missing file is ambiguous.

        :param text_path: Human-readable report destination.
        :param json_path: Optional machine-readable destination.
        """
        text_path.parent.mkdir(parents=True, exist_ok=True)
        grouped = self.by_stage()

        lines = ["# PDF / ingestion failures", ""]
        if not self.failures:
            lines.append("No failures recorded. All papers were fetched, parsed, and indexed.")
        else:
            lines.append(f"{len(self.failures)} paper(s) failed across {len(grouped)} stage(s).")
            lines.append("")
            for stage in sorted(grouped):
                stage_failures = grouped[stage]
                lines.append(f"## {stage} ({len(stage_failures)})")
                # Bare ID list first, so it can be copy-pasted straight into a retry.
                lines.append("arxiv_ids: " + ", ".join(f.arxiv_id for f in stage_failures))
                lines.append("")
                for failure in stage_failures:
                    lines.append(f"  - {failure.arxiv_id}: {failure.reason}")
                    if failure.title:
                        lines.append(f"      title: {failure.title[:100]}")
                lines.append("")

        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "failure_count": len(self.failures),
                "by_stage": {stage: [f.arxiv_id for f in items] for stage, items in grouped.items()},
                "failures": [asdict(f) for f in self.failures],
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
