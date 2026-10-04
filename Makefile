.PHONY: build build-all-arch push run seasonality

IMAGE ?= nherbaut/meal-planning:latest
ENV_FILE ?= .env

# Detect host arch and map to Docker platform
HOST_ARCH := $(shell uname -m)
ifeq ($(HOST_ARCH),x86_64)
PLATFORM := linux/amd64
else ifeq ($(HOST_ARCH),aarch64)
PLATFORM := linux/arm64
else
PLATFORM := linux/amd64
endif

build:
	DOCKER_BUILDKIT=1 docker buildx build --platform $(PLATFORM) -t $(IMAGE) --push .

build-all-arch:
	DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE) --push .

push:
	docker push $(IMAGE)

run:
	docker compose --env-file $(ENV_FILE) up -d --no-build --pull always

# Manual IA-1 pass; only new ingredients are submitted on later runs.
seasonality:
	docker compose --env-file $(ENV_FILE) build meal-planning
	docker compose --env-file $(ENV_FILE) run --rm --no-deps meal-planning python seasonality.py
