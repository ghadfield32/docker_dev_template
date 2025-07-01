from __future__ import annotations
import os
import socket
import logging
from pathlib import Path
from typing import Any, Sequence, Optional
from contextlib import closing

import mlflow
import psutil  # lightweight; already added to pyproject deps
from sklearn.utils.multiclass import type_of_target
from explainerdashboard import (
    ClassifierExplainer,
    RegressionExplainer,
    ExplainerDashboard,
)

logging.basicConfig(level=logging.INFO)

__all__ = ["build_and_log_dashboard", "load_dashboard_yaml", "_first_free_port", "_port_details"]


# ---------------------------------------------------------------------------
def _port_details(port: int) -> str:
    """
    Return a one-line string with PID & cmdline of the process
    listening on *port*, or '' if none / not discoverable.
    """
    for c in psutil.net_connections(kind="tcp"):
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
            try:
                p = psutil.Process(c.pid)
                return f"[PID {p.pid} – {p.name()}] cmd={p.cmdline()}"
            except psutil.Error:
                return f"[PID {c.pid}] (no detail)"
    return ""

def _first_free_port(start: int = 8050, tries: int = 50) -> int:
    """Return first free TCP port ≥ *start* on localhost."""
    for port in range(start, start + tries):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.05)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            # Port is in use, try next one
            continue
    raise RuntimeError("⚠️  No free ports found in range")

def _next_free_port(start: int = 8050, tries: int = 50) -> int:
    """Return the first free TCP port ≥ *start*. (Alias for backward compatibility)"""
    return _first_free_port(start, tries)

def _port_in_use(port: int) -> bool:
    """Check if a port is already in use on any interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.05)
        # Check both localhost and 0.0.0.0 to be thorough
        try:
            # First check localhost (127.0.0.1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
            # Also check if anything is bound to all interfaces
            if s.connect_ex(("0.0.0.0", port)) == 0:
                return True
        except (socket.gaierror, OSError):
            # If we can't connect, assume port is free
            pass
        return False


# ---------------------------------------------------------------------------
def build_and_log_dashboard(
    model: Any,
    X_test,                        # 2-D ndarray / DataFrame
    y_test,                        # 1-D labels or targets
    *,
    # ---- new kwargs mirrored from ExplainerDashboard ----------------------
    cats: Optional[Sequence[str]] = None,
    idxs: Optional[Sequence[Any]] = None,
    descriptions: Optional[dict[str, str]] = None,
    target: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    X_background=None,
    model_output: str = "probability",
    shap: str = "guess",
    shap_interaction: bool = True,
    simple: bool = False,
    mode: str = "external",        # inline · jupyterlab · external
    title: str = "Model Explainer",
    # ---- infra ------------------------------------------------------------
    run: mlflow.ActiveRun | None = None,
    port: int | None = None,
    serve: bool = False,
    save_yaml: bool = True,
    output_dir: os.PathLike | str | None = None,
) -> Path:
    """
    Build + log ExplainerDashboard, with improved port handling:

    • If *port* is None we auto-select the first free port ≥ 8050  
    • If *port* is occupied we print owner details & **abort** (caller decides)  
      -- avoids silent failure that confused you earlier.

    Returns
    -------
    Path to the saved ``dashboard.yaml``    (or the HTML file if save_yaml=False)
    """
    # 1️⃣ Pick correct explainer ------------------------------------------------
    problem = type_of_target(y_test)
    if problem in {"continuous", "continuous-multioutput"}:
        ExplainerCls = RegressionExplainer
        # RegressionExplainer doesn't support 'labels' or 'model_output' parameters
        explainer_kwargs = {
            "cats": cats,
            "idxs": idxs,
            "descriptions": descriptions,
            "target": target,
            "X_background": X_background,
            "shap": shap,
        }
    else:
        ExplainerCls = ClassifierExplainer
        # ClassifierExplainer supports all parameters
        explainer_kwargs = {
            "cats": cats,
            "idxs": idxs,
            "descriptions": descriptions,
            "target": target,
            "labels": labels,
            "X_background": X_background,
            "model_output": model_output,
            "shap": shap,
        }

    # Filter out None values to avoid issues
    explainer_kwargs = {k: v for k, v in explainer_kwargs.items() if v is not None}
    
    explainer = ExplainerCls(
        model,
        X_test,
        y_test,
        **explainer_kwargs
    )

    dash = ExplainerDashboard(
        explainer,
        title=title,
        shap_interaction=shap_interaction,
        simple=simple,
        mode=mode,
    )

    # 2️⃣ Persist + log artefacts ----------------------------------------------
    out_dir = Path(output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "explainer_dashboard.html"
    dash.save_html(html_path)
    mlflow.log_artifact(str(html_path))

    yaml_path: Path | None = None
    if save_yaml:
        yaml_path = out_dir / "dashboard.yaml"
        dash.to_yaml(yaml_path)
        mlflow.log_artifact(str(yaml_path))

    # 3️⃣ Optional serving ----------------------------------------------------
    if serve:
        chosen = port or _first_free_port()
        if _port_in_use(chosen):
            details = _port_details(chosen)
            raise RuntimeError(
                f"❌ Port {chosen} already in use {details}. "
                "Either pass a different --port or stop the process."
            )
        logging.info("🌐 Serving dashboard on http://0.0.0.0:%s", chosen)
        dash.run(chosen, host="0.0.0.0", use_waitress=True, open_browser=False)

    return yaml_path or html_path

# ---------------------------------------------------------------------------
def load_dashboard_yaml(path: os.PathLike | str) -> ExplainerDashboard:
    """Reload a YAML config – unchanged but kept for public API."""
    return ExplainerDashboard.from_config(path) 

   
