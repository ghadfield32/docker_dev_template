# tasks.py  ── invoke ≥2.2
from invoke import task, Context
from typing import List, Optional

import os
import sys
import pathlib
import tempfile
import datetime as _dt
import atexit


BASE_ENV = pathlib.Path(__file__).parent


# Track temporary env files for cleanup
_saved_env_files: List[str] = []


def _write_envfile(name: str, ports: Optional[dict[str, int]] = None) -> pathlib.Path:
    """Generate an .env file customised for this run & return its path."""
    env_lines = [f"ENV_NAME={name}"]
    mapping = {
        "jupyter": "HOST_JUPYTER_PORT",
        "tensorboard": "HOST_TENSORBOARD_PORT",
        "explainer": "HOST_EXPLAINER_PORT",
        "streamlit": "HOST_STREAMLIT_PORT",
    }
    for svc, var in mapping.items():
        if ports and svc in ports:
            env_lines.append(f"{var}={ports[svc]}")
    # fall back to template defaults for everything else
    env_lines.append(f"# generated {_dt.datetime.now().isoformat()}")
    tmp = tempfile.NamedTemporaryFile(
        "w", 
        delete=False, 
        prefix=".env.",
        dir=BASE_ENV
    )
    tmp.write("\n".join(env_lines))
    tmp.close()
    _saved_env_files.append(tmp.name)
    return pathlib.Path(tmp.name)


# Register cleanup function
def _cleanup_env_files() -> None:
    """Remove all temporary env files."""
    for path in _saved_env_files:
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_env_files)


def _compose(
    c: Context,
    cmd: str,
    name: str,
    rebuild: bool = False,
    force_pty: bool = False,
    ports: Optional[dict[str, int]] = None,
) -> None:
    """
    Wrapper around `docker compose` that

    • Injects ENV_NAME and COMPOSE_PROJECT_NAME so both build-time (*args*)
      and runtime (*docker-compose.yml* env) use one canonical name.
    • Passes `-p <n>` so images / volumes share that namespace.
    • Falls back gracefully when PTYs are unavailable (Windows CI).
    • Allows custom port configuration via ports dict.
    """
    env = {**os.environ, "ENV_NAME": name, "COMPOSE_PROJECT_NAME": name}
    
    # Add port overrides if provided
    if ports:
        port_mapping = {
            "jupyter": "HOST_JUPYTER_PORT",
            "tensorboard": "HOST_TENSORBOARD_PORT", 
            "explainer": "HOST_EXPLAINER_PORT",
            "streamlit": "HOST_STREAMLIT_PORT",
        }
        for service, port in ports.items():
            if service in port_mapping:
                env[port_mapping[service]] = str(port)
    
    use_pty = force_pty or (os.name != "nt" and sys.stdin.isatty())

    if not use_pty and not getattr(_compose, "_warned", False):
        print("ℹ️  PTY not supported – running without TTY.")
        _compose._warned = True  # type: ignore[attr-defined]

    if rebuild:
        full_cmd = f"docker compose -p {name} {cmd} --build --pull"
    else:
        full_cmd = f"docker compose -p {name} {cmd}"
    c.run(full_cmd, env=env, pty=use_pty)


@task(
    help={
        "name": "Project/venv name (defaults to folder name)",
        "use_pty": "Force PTY even on non-POSIX hosts",
        "jupyter_port": "Jupyter Lab port (default: 8890)",
        "tensorboard_port": "TensorBoard port (default: auto-assigned)",
        "explainer_port": "Explainer Dashboard port (default: auto-assigned)", 
        "streamlit_port": "Streamlit port (default: auto-assigned)",
    }
)
def up(
    c,
    name: Optional[str] = None,
    rebuild: bool = False,
    detach: bool = True,
    use_pty: bool = False,
    jupyter_port: Optional[int] = None,
    tensorboard_port: Optional[int] = None,
    explainer_port: Optional[int] = None,
    streamlit_port: Optional[int] = None,
) -> None:
    """Build (optionally --rebuild) & start the container with custom ports."""
    name = name or BASE_ENV.name
    
    # Build ports dict from provided arguments
    ports = {}
    if jupyter_port is not None:
        ports["jupyter"] = jupyter_port
    if tensorboard_port is not None:
        ports["tensorboard"] = tensorboard_port
    if explainer_port is not None:
        ports["explainer"] = explainer_port
    if streamlit_port is not None:
        ports["streamlit"] = streamlit_port
    
    # Generate environment file
    env_path = _write_envfile(name, ports)
    env_file_flag = f"--env-file {env_path}"
    compose_cmd = "up -d" if detach else "up"

    _compose(
        c,
        f"{env_file_flag} {compose_cmd}",
        name,
        rebuild=rebuild,
        force_pty=use_pty,
        ports=ports if ports else None,
    )


@task(
    help={
        "name": "Project/venv name (defaults to folder name)",
    }
)
def stop(c, name: Optional[str] = None) -> None:
    """Stop and remove dev container (keeps volumes)."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} down"
    try:
        c.run(cmd)
        print(f"\n🛑 Stopped and removed project '{name}'")
    except Exception:
        print(f"❌ No running containers found for project '{name}'")


@task
def shell(c, name: str | None = None) -> None:
    """Open an interactive shell inside the running container."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} ps -q datascience"
    cid = c.run(cmd, hide=True).stdout.strip()
    c.run(f"docker exec -it {cid} bash", env={"ENV_NAME": name}, pty=False)


@task
def clean(c) -> None:
    """Prune stopped containers + dangling images."""
    c.run("docker system prune -f")


@task
def ports(c, name: str | None = None) -> None:
    """Show current port mappings for the named project."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} ps --format table"
    try:
        c.run(cmd, hide=False)
        print(f"\n📊 Port mappings for project '{name}':")
        print("=" * 50)
    except Exception:
        print(f"❌ No running containers found for project '{name}'")
        print("\n💡 Usage examples:")
        print("  invoke up --name myproject --jupyter-port 8891")
        print("  invoke up --name myproject --jupyter-port 8892 \\")
        print("    --tensorboard-port 6009")
