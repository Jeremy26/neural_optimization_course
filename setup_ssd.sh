#!/usr/bin/env bash
# Configure /ssd (NVMe) as the default for all package caches and large files.
# Run as root on your actual machine: sudo bash setup_ssd.sh
#
# What this does:
#   1. Verifies /ssd is mounted (exits if not)
#   2. Creates the expected directory structure
#   3. Moves existing caches from root fs to /ssd, leaving symlinks behind
#   4. Writes /etc/profile.d/ssd-caches.sh (auto-loaded by every shell)
#   5. Writes pip.conf and uv.toml
#   6. Optionally adds noatime to the fstab entry if missing
#   7. Optionally moves conda package cache to /ssd
set -e

SSD=/ssd

# ── Preflight ─────────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo bash $0"
    exit 1
fi

if ! mountpoint -q "$SSD" 2>/dev/null; then
    echo "ERROR: $SSD is not a mountpoint."
    echo "Mount your NVMe first, e.g.:"
    echo "  sudo mount /dev/nvme0n1p1 /ssd"
    echo "Then re-run this script."
    exit 1
fi

echo "Using SSD at: $SSD ($(df -h "$SSD" | awk 'NR==2{print $2, "total,", $4, "free"}'))"

# ── Directory structure ───────────────────────────────────────────────────────
echo ""
echo "Creating directory structure..."
mkdir -p "$SSD"/{data,models,tmp,.jupyter}
mkdir -p "$SSD"/.cache/{pip,uv,torch,huggingface/hub,sentence_transformers,jupyter}
chmod 1777 "$SSD/tmp"
echo "  Done."

# ── Move caches from root fs ──────────────────────────────────────────────────
move_cache() {
    local src=$1 dst=$2
    if [ -d "$src" ] && [ ! -L "$src" ]; then
        echo "  Moving $src → $dst ..."
        cp -a "$src/." "$dst/"
        rm -rf "$src"
        ln -s "$dst" "$src"
        echo "  Symlink: $src -> $dst"
    elif [ -L "$src" ]; then
        echo "  $src is already a symlink (skipping)"
    else
        echo "  $src not found (nothing to move)"
    fi
}

echo ""
echo "Moving caches to /ssd..."
move_cache /root/.cache/uv           "$SSD/.cache/uv"
move_cache /root/.cache/pip          "$SSD/.cache/pip"
move_cache /root/.cache/torch        "$SSD/.cache/torch"
move_cache /root/.cache/huggingface  "$SSD/.cache/huggingface"

# Handle per-user cache dirs if running as a non-root user was common
for uhome in /home/*; do
    uname=$(basename "$uhome")
    move_cache "$uhome/.cache/pip"         "$SSD/.cache/pip"
    move_cache "$uhome/.cache/torch"       "$SSD/.cache/torch"
    move_cache "$uhome/.cache/huggingface" "$SSD/.cache/huggingface"
done

# ── /etc/profile.d ────────────────────────────────────────────────────────────
echo ""
echo "Writing /etc/profile.d/ssd-caches.sh..."
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
echo "  Done."

# ── pip config ────────────────────────────────────────────────────────────────
echo ""
echo "Writing pip.conf..."
mkdir -p /root/.config/pip
cat > /root/.config/pip/pip.conf <<'EOF'
[global]
cache-dir = /ssd/.cache/pip
EOF
echo "  Done: /root/.config/pip/pip.conf"

# ── uv config ─────────────────────────────────────────────────────────────────
echo ""
echo "Writing uv.toml..."
mkdir -p /root/.config/uv
cat > /root/.config/uv/uv.toml <<'EOF'
cache-dir = "/ssd/.cache/uv"
EOF
echo "  Done: /root/.config/uv/uv.toml"

# ── fstab: add noatime if missing ─────────────────────────────────────────────
echo ""
fstab_line=$(grep -E '\s/ssd[\s/]' /etc/fstab 2>/dev/null | head -1)
if [ -n "$fstab_line" ]; then
    if echo "$fstab_line" | grep -q "noatime"; then
        echo "fstab: noatime already present."
    else
        echo "fstab: adding noatime to /ssd entry..."
        cp /etc/fstab /etc/fstab.bak
        # Insert noatime after the options field (4th column)
        sed -i "s|\(.*\s/ssd\s.*\)\(defaults\)|\1\2,noatime|" /etc/fstab
        echo "  Backup saved to /etc/fstab.bak"
        echo "  Updated entry:"
        grep -E '\s/ssd[\s/]' /etc/fstab
    fi
else
    echo "WARNING: No /ssd entry found in /etc/fstab."
    echo "  Add one to survive reboots, e.g.:"
    dev=$(findmnt -n -o SOURCE /ssd 2>/dev/null)
    uuid=$(blkid -s UUID -o value "$dev" 2>/dev/null)
    fstype=$(findmnt -n -o FSTYPE /ssd 2>/dev/null)
    if [ -n "$uuid" ]; then
        echo "  UUID=$uuid  /ssd  $fstype  defaults,noatime  0  2"
    else
        echo "  <device>  /ssd  <fstype>  defaults,noatime  0  2"
    fi
fi

# ── Conda package cache ───────────────────────────────────────────────────────
echo ""
if command -v conda &>/dev/null; then
    conda_pkgs=$(conda info --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['pkgs_dirs'][0])" 2>/dev/null)
    if [[ "$conda_pkgs" != /ssd* ]]; then
        echo "Configuring conda package cache → /ssd/.cache/conda..."
        mkdir -p "$SSD/.cache/conda"
        conda config --system --add pkgs_dirs "$SSD/.cache/conda"
        echo "  Done. Current pkgs_dirs:"
        conda config --show pkgs_dirs
    else
        echo "conda pkgs_dirs already on /ssd."
    fi
else
    echo "conda not installed — skipping."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete."
echo "  Activate now:  source /etc/profile.d/ssd-caches.sh"
echo "  Verify with:   bash health_check_ssd.sh"
echo "══════════════════════════════════════════════"
echo ""
echo "/ssd disk usage:"
df -h "$SSD"
echo ""
echo "/ssd cache sizes:"
du -sh "$SSD"/.cache/* 2>/dev/null | sort -rh
echo ""
