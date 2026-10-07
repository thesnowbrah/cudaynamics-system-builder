from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .generator import GeneratedProject
from .model import SystemSpec


class InstallError(RuntimeError):
    pass


@dataclass(slots=True)
class InstallPlan:
    writes: dict[Path, str]
    diff: str


def locate_default_project(builder_root: Path) -> Path:
    candidate = builder_root.parent / "cudaynamics-master"
    return candidate if candidate.is_dir() else builder_root.parent


def _insert_before_group_end(text: str, marker: str, addition: str) -> str:
    position = text.find(marker)
    if position < 0:
        raise InstallError(f"В файле проекта не найден маркер {marker!r}")
    end = text.find("</ItemGroup>", position)
    if end < 0:
        raise InstallError("Повреждён XML проекта: нет </ItemGroup>")
    return text[:end] + addition + text[end:]


def build_plan(project_root: str | Path, spec: SystemSpec, generated: GeneratedProject) -> InstallPlan:
    root = Path(project_root).resolve()
    required = ["systemsHeaders.h", "main.cpp", "CUDAynamics.vcxproj", "CUDAynamics.vcxproj.filters", "systems"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise InstallError("Это не корень CUDAynamics. Не найдены: " + ", ".join(missing))
    target_dir = root / "systems" / spec.system_id
    if target_dir.exists():
        raise InstallError(f"Система {spec.system_id!r} уже существует")

    originals = {name: (root / name).read_text(encoding="utf-8-sig") for name in required[:-1]}
    sid = spec.system_id
    updated: dict[str, str] = {}

    headers = originals["systemsHeaders.h"]
    include = f'#include "systems/{sid}/{sid}.h"'
    if include in headers:
        raise InstallError("Заголовок системы уже зарегистрирован")
    updated["systemsHeaders.h"] = headers.rstrip() + f"\n{include}\n"

    main = originals["main.cpp"]
    anchor = "    //selectKernel(kernels.begin()->first);"
    if anchor not in main:
        anchor = "    selectKernel(lorenz);"
    if anchor not in main:
        raise InstallError("Не найдено место регистрации addKernel в main.cpp")
    updated["main.cpp"] = main.replace(anchor, f"    addKernel({sid});\n{anchor}", 1)

    vcx = originals["CUDAynamics.vcxproj"]
    updated["CUDAynamics.vcxproj"] = _insert_before_group_end(
        vcx, '<CudaCompile Include="systems\\', f'    <CudaCompile Include="systems\\{sid}\\{sid}.cu" />\n  '
    )
    filters = originals["CUDAynamics.vcxproj.filters"]
    updated["CUDAynamics.vcxproj.filters"] = _insert_before_group_end(
        filters, '<CudaCompile Include="systems\\', f'    <CudaCompile Include="systems\\{sid}\\{sid}.cu" />\n  '
    )

    writes: dict[Path, str] = {root / name: content for name, content in updated.items()}
    for relative, content in generated.files.items():
        writes[root / Path(relative)] = content

    diffs = []
    for name, new in updated.items():
        diffs.extend(difflib.unified_diff(originals[name].splitlines(), new.splitlines(), fromfile=name, tofile=name, lineterm=""))
    for relative, content in generated.files.items():
        diffs.extend(difflib.unified_diff([], content.splitlines(), fromfile="/dev/null", tofile=relative, lineterm=""))
    return InstallPlan(writes, "\n".join(diffs))


def apply_plan(project_root: str | Path, plan: InstallPlan) -> Path:
    root = Path(project_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = root / ".cudynamics-builder-backups" / stamp
    changed_existing = [path for path in plan.writes if path.exists()]
    created = [path for path in plan.writes if not path.exists()]
    try:
        for path in changed_existing:
            relative = path.relative_to(root)
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for path, content in plan.writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".builder-tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
    except Exception as exc:
        for path in created:
            if path.exists():
                path.unlink()
        for path in changed_existing:
            saved = backup / path.relative_to(root)
            if saved.exists():
                shutil.copy2(saved, path)
        raise InstallError(f"Установка отменена и изменения откачены: {exc}") from exc
    return backup

