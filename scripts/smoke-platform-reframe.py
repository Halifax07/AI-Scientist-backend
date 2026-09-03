"""Smoke test the presets + ProjectSpec defaults after the platform reframe."""
import json
from fsad_scientist.domain.presets import load_preset_registry, resolve_preset
from fsad_scientist.domain.models import ProjectSpec


def main() -> None:
    registry = load_preset_registry()
    print("registered preset ids:", sorted(registry.keys()))
    mvad = registry.get("machine_vision_anomaly_detection")
    assert mvad is not None, "machine_vision_anomaly_detection preset must be registered"
    print("default_title:", mvad.default_title)
    print("default_domain:", mvad.default_domain)
    print("default_keywords:", mvad.default_keywords)

    fsad = registry["fsad"]
    assert fsad.label.startswith("少样本工业视觉异常检测")

    generic = resolve_preset(None)
    print("resolve_preset(None) ->", generic.id)

    spec = ProjectSpec()
    print("ProjectSpec default preset:", spec.preset)
    print("ProjectSpec default title:", spec.title)
    print("ProjectSpec default domain:", spec.domain)
    print("ProjectSpec default application_context:", spec.application_context)


if __name__ == "__main__":
    main()
