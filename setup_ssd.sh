#!/usr/bin/env bash
# Run once on a new machine to route all large caches and downloads to /ssd.
# If /ssd is a real mount, run this after mounting. Otherwise /ssd will be a
# directory on the root filesystem acting as a staging area.
set -e

SSD=/ssd

# Directory layout
mkdir -p "$SSD"/{data,models,tmp,.cache/{pip,uv,torch,huggingface,jupyter},.jupyter}
chmod 1777 "$SSD/tmp"

# Move existing caches that are already on root (if present)
for cache in pip uv; do
    src="/root/.cache/$cache"
    dst="$SSD/.cache/$cache"
    if [ -d "$src" ] && [ ! -L "$src" ]; then
        cp -a "$src/." "$dst/"
        rm -rf "$src"
        ln -s "$dst" "$src"
        echo "Moved $src → $dst"
    fi
done

# System-wide env vars — sourced by every login shell
cat > /etc/profile.d/ssd-caches.sh <<'EOF'
# Route all large caches and downloads to /ssd to keep the root filesystem free.
export UV_CACHE_DIR=/ssd/.cache/uv
export PIP_CACHE_DIR=/ssd/.cache/pip
export TORCH_HOME=/ssd/.cache/torch
export HF_HOME=/ssd/.cache/huggingface
export TRANSFORMERS_CACHE=/ssd/.cache/huggingface/hub
export SENTENCE_TRANSFORMERS_HOME=/ssd/.cache/sentence_transformers
export JUPYTER_DATA_DIR=/ssd/.jupyter
export TMPDIR=/ssd/tmp
EOF
chmod 644 /etc/profile.d/ssd-caches.sh

# pip config
mkdir -p /root/.config/pip
cat > /root/.config/pip/pip.conf <<'EOF'
[global]
cache-dir = /ssd/.cache/pip
EOF

# uv config
mkdir -p /root/.config/uv
cat > /root/.config/uv/uv.toml <<'EOF'
cache-dir = "/ssd/.cache/uv"
EOF

echo ""
echo "Done. Source the profile now with:"
echo "  source /etc/profile.d/ssd-caches.sh"
echo ""
echo "/ssd layout:"
du -sh "$SSD"/* "$SSD"/.*/ 2>/dev/null | grep -v "^\.\.$" | sort -rh
