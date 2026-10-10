CROSS_CXX ?= aarch64-unknown-linux-musl-c++
QEMU ?= qemu-aarch64
CXXFLAGS ?= -O3 -g -std=c++17 -march=armv9-a+sme2 -Wall -Wextra -Werror
LDFLAGS ?= -static
SVL_BYTES ?= 16 32 64 128 256

.PHONY: all check benchmark benchmark-smoke clean
all: build/test build/benchmark

build/%: %.cpp gated_delta.h Makefile
	mkdir -p build
	$(CROSS_CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o "$@"

check: build/test
	@set -eu; for bytes in $(SVL_BYTES); do \
		echo "SME streaming vector length: $$bytes bytes"; \
		$(QEMU) -cpu max,sme-default-vector-length=$$bytes ./build/test; \
	done

benchmark: build/benchmark

benchmark-smoke: build/benchmark
	@echo "QEMU smoke test only: these timings do not represent C1 performance."
	$(QEMU) -cpu max,sme-default-vector-length=64 ./build/benchmark 32 3

clean:
	rm -rf build
