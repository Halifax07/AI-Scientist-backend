from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fsad_scientist.domain.models import ResearchProject


class ProjectNotFoundError(KeyError):
    pass


class JsonProjectRepository:
    """Small durable Research Ledger used before PostgreSQL is introduced.

    Each project is stored in its own directory so future experiment artifacts can
    be colocated without turning one global JSON document into a write bottleneck.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, project: ResearchProject) -> ResearchProject:
        project_dir = self.root / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        destination = project_dir / "project.json"
        serialized = project.model_dump_json(indent=2)

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=project_dir,
            prefix="project-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temp_path = Path(handle.name)

        os.replace(temp_path, destination)
        return project

    def get(self, project_id: str) -> ResearchProject:
        path = self.root / project_id / "project.json"
        if not path.exists():
            raise ProjectNotFoundError(project_id)
        return ResearchProject.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ResearchProject]:
        projects: list[ResearchProject] = []
        for path in sorted(self.root.glob("*/project.json")):
            projects.append(
                ResearchProject.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

