# tasks.py  ── invoke ≥2.2
from invoke import task, Context
import os, shutil, sys, pathlib, textwrap, subprocess

BASE_ENV = pathlib.Path(__file__).parent

def _compose(c: Context, cmd: str, name: str, rebuild=False):
    # Propagate ENV_NAME into docker-compose *and* docker build-args
    env = dict(os.environ, ENV_NAME=name)
    flags = ["--build", "--pull"] if rebuild else []
    c.run(f"docker compose {' '.join(flags)} {cmd}", env=env, pty=True)

@task(help={'name': "Project/venv name (defaults to folder name)"},
      iterable=['extra'])
def up(c, name=None, rebuild=False, detach=True, extra=None):
    """Build (optionally --rebuild) & start the container"""
    name = name or BASE_ENV.name
    _compose(c, "up -d" if detach else "up", name, rebuild)

@task
def stop(c):
    """Stop and remove dev container (keeps volumes)"""
    c.run("docker compose down")

@task
def shell(c, name=None):
    """Open an interactive shell in the running container"""
    name = name or BASE_ENV.name
    cid = c.run("docker compose ps -q datascience", hide=True).stdout.strip()
    c.run(f"docker exec -it {cid} bash", env={"ENV_NAME": name})

@task
def clean(c):
    """Prune stopped containers + dangling images"""
    c.run("docker system prune -f")
