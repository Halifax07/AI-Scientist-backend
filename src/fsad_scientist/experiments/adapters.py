from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.experiments.models import CommandSpec


class MethodAdapter(ABC):
    name: str

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    @abstractmethod
    def build_command(
        self,
        run: ExperimentRun,
        *,
        dataset_view: Path,
        output_dir: Path,
        device: str,
    ) -> CommandSpec: ...


class PatchCoreAdapter(MethodAdapter):
    name = "patchcore"

    def build_command(
        self,
        run: ExperimentRun,
        *,
        dataset_view: Path,
        output_dir: Path,
        device: str,
    ) -> CommandSpec:
        gpu_id = device.removeprefix("cuda:") if device.startswith("cuda:") else "0"
        return CommandSpec(
            method=self.name,
            executable="python",
            cwd=self.repository_root,
            environment={"PYTHONPATH": str(self.repository_root / "src")},
            args=[
                "bin/run_patchcore.py",
                "--gpu",
                gpu_id,
                "--seed",
                str(run.seed),
                "--save_patchcore_model",
                "--log_group",
                run.id,
                "--log_project",
                "fsad_scientist",
                str(output_dir),
                "patch_core",
                "-b",
                "wideresnet50",
                "-le",
                "layer2",
                "-le",
                "layer3",
                "--pretrain_embed_dimension",
                "1024",
                "--target_embed_dimension",
                "1024",
                "--anomaly_scorer_num_nn",
                "1",
                "--patchsize",
                "3",
                "sampler",
                "-p",
                "0.1",
                "approx_greedy_coreset",
                "dataset",
                "--resize",
                "256",
                "--imagesize",
                "224",
                "-d",
                run.category,
                "mvtec",
                str(dataset_view),
            ],
            expected_outputs=["**/results.csv", "**/models/**"],
            notes=[
                "dataset_view 必须只暴露预注册 support manifest 中的正常训练图像。",
                "当前命令保留官方 PatchCore 参数，少样本视图由本项目负责生成。",
            ],
        )


class AnomalyDinoAdapter(MethodAdapter):
    name = "anomalydino"

    def build_command(
        self,
        run: ExperimentRun,
        *,
        dataset_view: Path,
        output_dir: Path,
        device: str,
    ) -> CommandSpec:
        dataset_name = "VisA" if run.dataset.lower() == "visa" else "MVTec"
        project_root = self.repository_root.parents[2]
        project_source = project_root / "backend" / "src"
        if not project_source.is_dir():
            project_source = project_root / "src"
        return CommandSpec(
            method=self.name,
            executable="python",
            cwd=output_dir,
            args=[
                "-m",
                "fsad_scientist.integrations.anomalydino_single",
                "--repository",
                str(self.repository_root),
                "--category",
                run.category,
                "--",
                "--dataset",
                dataset_name,
                "--data_root",
                str(dataset_view),
                "--shots",
                str(run.shots),
                "--num_seeds",
                "1",
                "--just_seed",
                str(run.seed),
                "--preprocess",
                "agnostic",
                "--eval_segm",
                "--faiss_on_cpu",
                "--device",
                device,
                "--tag",
                run.id,
            ],
            environment={
                "PYTHONPATH": os.pathsep.join(
                    [str(project_source), str(self.repository_root)]
                )
            },
            expected_outputs=["**/metrics_seed=*.json", "**/plots/**"],
            notes=[
                "项目侧包装器将官方固定类别列表限制为当前 Run 的单个类别。",
                "K-only 数据视图走官方 full-shot 读取分支，避免 seed>0 的块切片返回空集。",
                "Windows 使用 faiss-cpu；视觉骨干仍在 CUDA 上运行。",
                "输出工作目录隔离在当前 Run 下；third_party 源码保持不变。",
            ],
        )


class SubspaceAdAdapter(MethodAdapter):
    name = "subspacead"

    def build_command(
        self,
        run: ExperimentRun,
        *,
        dataset_view: Path,
        output_dir: Path,
        device: str,
    ) -> CommandSpec:
        dataset_name = "visa" if run.dataset.lower() == "visa" else "mvtec_ad"
        return CommandSpec(
            method=self.name,
            executable="python",
            cwd=self.repository_root,
            args=[
                "main.py",
                "--seed",
                str(run.seed),
                "--dataset_name",
                dataset_name,
                "--dataset_path",
                str(dataset_view),
                "--categories",
                run.category,
                "--model_ckpt",
                "facebook/dinov2-with-registers-giant",
                "--image_res",
                "672",
                "--k_shot",
                str(run.shots),
                "--aug_count",
                "30",
                "--pca_ev",
                "0.99",
                "--outdir",
                str(output_dir),
            ],
            environment={"CUDA_VISIBLE_DEVICES": _device_index(device)},
            expected_outputs=["**/*.json", "**/*.csv", "**/*.png"],
            notes=[
                "显式传入 seed、类别和输出目录。",
                "正式实验需确认官方脚本的 support 抽样与本项目 manifest 完全一致。",
            ],
        )


class MethodRegistry:
    def __init__(self, project_root: Path) -> None:
        third_party = project_root / "third_party" / "anomaly_detection"
        self._adapters: dict[str, MethodAdapter] = {
            "patchcore": PatchCoreAdapter(third_party / "patchcore-inspection"),
            "anomalydino": AnomalyDinoAdapter(third_party / "anomalydino"),
            "subspacead": SubspaceAdAdapter(third_party / "subspacead"),
        }

    def get(self, name: str) -> MethodAdapter:
        try:
            return self._adapters[name.lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown anomaly detector adapter: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._adapters)


def _device_index(device: str) -> str:
    return device.removeprefix("cuda:") if device.startswith("cuda:") else ""
