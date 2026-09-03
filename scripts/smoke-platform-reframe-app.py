"""Confirm the FastAPI app boots with the refactored presets."""
from fsad_scientist.api.app import create_app


def main() -> None:
    app = create_app()
    routes = sorted({route.path for route in app.routes if hasattr(route, "path")})
    print("loaded app with", len(routes), "routes")
    print("preset/demo endpoints:")
    for path in routes:
        if "/projects" in path or "/presets" in path:
            print("  -", path)


if __name__ == "__main__":
    main()
