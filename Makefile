
##@ Development
.PHONY: help debug-interpreter monitor-resources check-health clean-rebuild

help: ## Display available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Debugging
debug-interpreter: ## Run systematic interpreter debugging (§7 from guide)
	@echo "🔍 Running Python interpreter diagnostics..."
	@./.devcontainer/debug_interpreter.sh

monitor-resources: ## Monitor container resources in real-time
	@echo "📊 Starting resource monitor (Ctrl+C to stop)..."
	@./.devcontainer/monitor_resources.sh

check-health: ## Check container health status
	@echo "🏥 Container health check..."
	@docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
	@echo ""
	@CONTAINER_ID=$$(docker ps --filter "ancestor=docker_dev_template-datascience:latest" --format "{{.ID}}" | head -1); if [[ -n "$$CONTAINER_ID" ]]; then echo "Restart count: $$(docker inspect -f '{{.State.RestartCount}}' $$CONTAINER_ID)"; else echo "No container found"; fi

##@ Container Management  
clean-rebuild: ## Clean rebuild of container (§5 from guide)
	@echo "🧹 Clean rebuilding container..."
	docker compose build --no-cache --progress=plain

##@ Quick Tests
test-python: ## Test Python environment inside container
	@echo "🐍 Testing Python environment..."
	@docker compose exec datascience bash -c "echo 'PATH=$$PATH' && which python && python -V && uv pip check"

test-imports: ## Test critical imports (JAX, PyTorch)
	@echo "📦 Testing critical imports..."
	@docker compose exec datascience python -c "import jax, torch; print('✅ JAX:', jax.__version__, '✅ PyTorch:', torch.__version__)"

##@ VS Code Integration
show-logs: ## Show VS Code dev container logs (requires container to be running)
	@echo "💡 To view VS Code logs:"
	@echo "  1. Press F1 in VS Code"
	@echo "  2. Run: 'Remote-Containers: Show Log'"
	@echo "  3. Check Output → Python and Output → Jupyter panels" 
