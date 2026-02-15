def normalize_cam_method(cam_method, supported_methods=None, default_for_aeromixer="RITSM"):
    """Normalize CAM method names and resolve aliases (including AEROMIXER)."""
    if cam_method is None:
        return ""

    method = str(cam_method).strip()
    if not method:
        return ""

    key = method.lower().replace("-", "").replace("_", "")
    aliases = {
        "ritsm": "RITSM",
        "ritsmclip": "RITSM",
        "ritsmclipvip": "RITSM",
        "hilacam": "HilaCAM",
        "hila": "HilaCAM",
        "mhsa": "MHSA",
        "mhsacam": "MHSA",
        "aeromixer": default_for_aeromixer,
        "auto": default_for_aeromixer,
        "default": default_for_aeromixer,
    }
    canonical = aliases.get(key, method)

    if supported_methods is None or canonical == "":
        return canonical

    if canonical not in supported_methods:
        options = ", ".join(sorted(supported_methods))
        raise ValueError(f"Unsupported CAM method '{cam_method}'. Supported methods: {options}")

    return canonical
