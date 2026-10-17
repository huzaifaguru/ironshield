#!/usr/bin/env bash
# Set up and start IronShield on an IBM LinuxONE Community Cloud server (s390x).
#
#   bash deploy/setup_linuxone.sh            # native Python (default)
#   bash deploy/setup_linuxone.sh docker     # IBM's prebuilt s390x ML image (has pandas, scikit-learn, XGBoost)
#
# Run from the repository root as the linux1 user.
set -euo pipefail
MODE="${1:-native}"
PORT="${IRONSHIELD_PORT:-8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== IronShield setup on $(uname -m) ($(. /etc/os-release && echo "$PRETTY_NAME")) =="

open_port() {
  echo "== Opening TCP port $PORT in iptables =="
  sudo iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null || \
    sudo iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT
  for f in /etc/iptables/rules.v4 /etc/sysconfig/iptables.save /etc/linuxone/iptables.save; do
    if [ -f "$f" ]; then sudo bash -c "iptables-save > $f"; echo "   saved rules to $f"; fi
  done
}

if [ "$MODE" = "native" ]; then
  # PyPI has no prebuilt s390x wheels for the newest numpy/scikit-learn (pip would compile for an hour),
  # so use the distribution's prebuilt s390x packages, including XGBoost.
  echo "== Installing prebuilt s390x Python packages =="
  if command -v apt-get >/dev/null; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-numpy python3-pandas \
      python3-sklearn python3-networkx python3-fastapi python3-uvicorn python3-requests python3-pytest curl
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-xgboost || true
  elif command -v dnf >/dev/null; then
    sudo dnf install -y python3 python3-pip python3-numpy python3-pandas python3-scikit-learn python3-networkx
  elif command -v zypper >/dev/null; then
    sudo zypper --non-interactive install python3 python3-pip python3-numpy python3-pandas python3-scikit-learn
  fi

  echo "== Creating virtual environment (reusing system packages) =="
  python3 -m venv --system-site-packages .venv
  . .venv/bin/activate
  python -c "import fastapi, uvicorn, networkx" 2>/dev/null || pip install networkx fastapi uvicorn requests
  BACKEND=xgboost
  if ! python -c "import xgboost" 2>/dev/null; then
    echo "!! xgboost is not available on this platform; using the scikit-learn backend"
    BACKEND=sklearn
  fi
  python -c "import platform, numpy, sklearn; print('python', platform.python_version(), 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'on', platform.machine())"

  echo "== Training the model on LinuxONE =="
  python -m scripts.train --backend "$BACKEND"
  echo "== Benchmarking on LinuxONE =="
  python -m scripts.benchmark --n 20000

  open_port
  echo "== Starting the API on port $PORT (log: api.log) =="
  # systemd service: keeps the API running across SSH disconnects and reboots
  sudo tee /etc/systemd/system/ironshield.service >/dev/null <<UNIT
[Unit]
Description=IronShield fraud detection API
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python -m uvicorn ironshield.api:app --host 0.0.0.0 --port $PORT
Restart=always
StandardOutput=append:$ROOT/api.log
StandardError=append:$ROOT/api.log

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now ironshield
  for i in $(seq 1 30); do curl -s "localhost:$PORT/health" && break; sleep 2; done; echo

elif [ "$MODE" = "docker" ]; then
  IMAGE=registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
  if ! command -v docker >/dev/null; then
    echo "== Installing Docker =="
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker "$USER"
    sudo systemctl start docker
  fi
  sudo docker pull "$IMAGE"
  open_port
  sudo docker rm -f ironshield >/dev/null 2>&1 || true
  echo "== Starting IronShield inside IBM's s390x ML image =="
  sudo docker run -d --name ironshield --network host -u root -v "$ROOT":/app -w /app --entrypoint bash "$IMAGE" -c \
    "pip install networkx fastapi uvicorn requests && python -m scripts.train && python -m scripts.benchmark --n 20000 \
     && uvicorn ironshield.api:app --host 0.0.0.0 --port $PORT"
  echo "Follow progress with: sudo docker logs -f ironshield"
else
  echo "Unknown mode '$MODE' (use native or docker)"; exit 1
fi

IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')
echo
echo "IronShield API: http://$IP:$PORT  (docs at /docs)"
echo "On your laptop:  set IRONSHIELD_API=http://$IP:$PORT  then  streamlit run dashboard/app.py"
