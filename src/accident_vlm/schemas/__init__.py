_EXPORT_MODULES = {
    "AccidentFactOutput": "accident_vlm.schemas.final_output",
    "AccidentType": "accident_vlm.schemas.final_output",
    "Confidence": "accident_vlm.schemas.common",
    "EvidenceField": "accident_vlm.schemas.common",
    "InputQuality": "accident_vlm.schemas.preprocessing",
    "PipelineContext": "accident_vlm.schemas.preprocessing",
    "SceneType": "accident_vlm.schemas.final_output",
    "SelectedFrame": "accident_vlm.schemas.preprocessing",
    "Status": "accident_vlm.schemas.common",
    "VideoMetadata": "accident_vlm.schemas.preprocessing",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
