# ==============================================================================
# Docling ExApp for Nextcloud - Makefile
# ==============================================================================

.PHONY: help build push dev test lint format clean

# Configuration
APP_ID := docling
APP_VERSION := 1.0.0
REGISTRY := ghcr.io
IMAGE_NAME := $(REGISTRY)/your-org/docling-nextcloud
NEXTCLOUD_URL ?= http://localhost:8080

# Default target
help:
	@echo "Docling ExApp for Nextcloud"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  dev          - Run locally for development"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  build        - Build Docker image"
	@echo "  build-dev    - Build development Docker image"
	@echo "  push         - Push image to registry"
	@echo ""
	@echo "Nextcloud Integration:"
	@echo "  register     - Register ExApp with Nextcloud"
	@echo "  unregister   - Unregister ExApp from Nextcloud"
	@echo "  enable       - Enable the ExApp"
	@echo "  disable      - Disable the ExApp"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean build artifacts"
	@echo "  logs         - Show container logs"

# ==============================================================================
# Development
# ==============================================================================

dev:
	@echo "Starting development server..."
	cd ex_app/lib && python -m uvicorn main:APP --host 0.0.0.0 --port 9000 --reload

test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Running linters..."
	ruff check ex_app/lib/
	mypy ex_app/lib/

format:
	@echo "Formatting code..."
	black ex_app/lib/
	ruff check --fix ex_app/lib/

# ==============================================================================
# Docker
# ==============================================================================

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(APP_VERSION) -t $(IMAGE_NAME):latest .

build-dev:
	@echo "Building development Docker image..."
	docker build --target builder -t $(IMAGE_NAME):dev .

push:
	@echo "Pushing Docker image to registry..."
	docker push $(IMAGE_NAME):$(APP_VERSION)
	docker push $(IMAGE_NAME):latest

run:
	@echo "Running Docker container..."
	docker run -it --rm \
		-p 9000:9000 \
		-e APP_ID=$(APP_ID) \
		-e APP_VERSION=$(APP_VERSION) \
		-e APP_SECRET=dev_secret_key \
		-e APP_HOST=0.0.0.0 \
		-e APP_PORT=9000 \
		-v docling_cache:/app/cache \
		$(IMAGE_NAME):latest

# ==============================================================================
# Nextcloud Integration
# ==============================================================================

register:
	@echo "Registering ExApp with Nextcloud..."
	@echo "Make sure AppAPI is installed and a Deploy Daemon is configured."
	curl -X POST "$(NEXTCLOUD_URL)/ocs/v2.php/apps/app_api/api/v1/ex-app" \
		-H "OCS-APIREQUEST: true" \
		-H "Content-Type: application/json" \
		-u admin:admin \
		-d '{"appid": "$(APP_ID)", "daemon_config_name": "docker_local"}'

unregister:
	@echo "Unregistering ExApp from Nextcloud..."
	curl -X DELETE "$(NEXTCLOUD_URL)/ocs/v2.php/apps/app_api/api/v1/ex-app?appid=$(APP_ID)" \
		-H "OCS-APIREQUEST: true" \
		-u admin:admin

enable:
	@echo "Enabling ExApp..."
	curl -X PUT "$(NEXTCLOUD_URL)/ocs/v2.php/apps/app_api/api/v1/ex-app/$(APP_ID)/enabled" \
		-H "OCS-APIREQUEST: true" \
		-H "Content-Type: application/json" \
		-u admin:admin \
		-d '{"enabled": true}'

disable:
	@echo "Disabling ExApp..."
	curl -X PUT "$(NEXTCLOUD_URL)/ocs/v2.php/apps/app_api/api/v1/ex-app/$(APP_ID)/enabled" \
		-H "OCS-APIREQUEST: true" \
		-H "Content-Type: application/json" \
		-u admin:admin \
		-d '{"enabled": false}'

# ==============================================================================
# Utilities
# ==============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf ex_app/lib/__pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

logs:
	@echo "Showing container logs..."
	docker logs -f $(APP_ID)

shell:
	@echo "Opening shell in container..."
	docker exec -it $(APP_ID) /bin/bash

