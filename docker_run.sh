#!/bin/bash
# ==============================================================================
# Helper Script for Coffee Bean Edge AI Docker on Jetson
# ==============================================================================

ACTION=${1:-"help"}

case "$ACTION" in
    build)
        echo "[*] Building Docker Image on Jetson ARM64..."
        docker compose build
        ;;
    up|server)
        echo "[*] Starting Coffee Edge AI Web Server on http://0.0.0.0:8000..."
        docker compose up -d
        echo "[+] Container running! Open browser: http://192.168.55.1:8000"
        ;;
    down|stop)
        echo "[*] Stopping Container..."
        docker compose down
        ;;
    benchmark)
        echo "[*] Running Terminal Benchmark inside Docker Container..."
        docker compose run --rm coffee-edge-ai python benchmarking/run_all_benchmarks.py
        ;;
    logs)
        docker compose logs -f
        ;;
    shell)
        docker compose run --rm coffee-edge-ai /bin/bash
        ;;
    *)
        echo "Usage: ./docker_run.sh [build | up | stop | benchmark | logs | shell]"
        echo "  build     : Build Docker image locally on Jetson"
        echo "  up        : Run Web Dashboard in background (port 8000)"
        echo "  stop      : Stop Web Dashboard"
        echo "  benchmark : Run accuracy preservation & latency benchmark inside container"
        echo "  logs      : View live container logs"
        echo "  shell     : Enter container terminal"
        ;;
esac
