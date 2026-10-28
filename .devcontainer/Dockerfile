FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Kinesis does not use Yarn. Its inherited package source has a stale
# signing key and otherwise prevents installation of container features.
# Disable that source while preserving package signature verification.
RUN if [ -f /etc/apt/sources.list.d/yarn.list ]; then \
      mv /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.disabled; \
    fi
