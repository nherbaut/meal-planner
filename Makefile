.PHONY: build build-all-arch run

IMAGE ?= nherbaut/meal-planning:latest

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

run:
	DOCKER_BUILDKIT=1 docker compose up -d --no-build
