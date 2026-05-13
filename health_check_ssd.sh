#!/usr/bin/env bash
# Health check for /ssd (NVMe) setup.
# Run on your actual machine: bash health_check_ssd.sh

PASS=0; WARN=0; FAIL=0
ok()   { echo "  [OK]   $*"; ((PASS++)); }
warn() { echo "  [WARN] $*"; ((WARN++)); }
fail() { echo "  [FAIL] $*"; ((FAIL++)); }

echo ""
echo "══════════════════════════════════════════════"
echo "  /ssd Health Check"
echo "══════════════════════════════════════════════"

# ── 1. NVMe device present ────────────────────────
echo ""
echo "── Block devices ────────────────────────────"
lsblk -o NAME,TYPE,FSTYPE,SIZE,MOUNTPOINT,MODEL 2>/dev/null || lsblk
nvme_devs=$(ls /dev/nvme* 2>/dev/null)
if [ -n "$nvme_devs" ]; then
    ok "NVMe device(s) found: $nvme_devs"
else
    fail "No NVMe devices found under /dev/nvme*"
fi

# ── 2. /ssd mounted ───────────────────────────────
echo ""
echo "── Mount status ─────────────────────────────"
if mountpoint -q /ssd 2>/dev/null; then
    ok "/ssd is mounted"
    mount | grep " /ssd "
    df -h /ssd
else
    fail "/ssd is NOT a mountpoint"
fi

# ── 3. fstab entry ────────────────────────────────
echo ""
echo "── /etc/fstab ───────────────────────────────"
fstab_entry=$(grep -E '\s/ssd\s' /etc/fstab 2>/dev/null)
if [ -n "$fstab_entry" ]; then
    ok "fstab entry found:"
    echo "       $fstab_entry"
    # Check for noatime (important for SSD longevity)
    if echo "$fstab_entry" | grep -q "noatime"; then
        ok "noatime option present"
    else
        warn "noatime not set — consider adding it for SSD longevity"
    fi
else
    fail "/ssd has no entry in /etc/fstab — mount will not survive reboot"
fi

# ── 4. Filesystem type ────────────────────────────
echo ""
echo "── Filesystem ───────────────────────────────"
fstype=$(findmnt -n -o FSTYPE /ssd 2>/dev/null)
if [ -n "$fstype" ]; then
    ok "Filesystem type: $fstype"
    case "$fstype" in
        ext4|xfs|btrfs|f2fs) ok "$fstype is a good choice for NVMe" ;;
        *) warn "Unusual filesystem: $fstype" ;;
    esac
else
    warn "Could not determine filesystem type (is /ssd mounted?)"
fi

# ── 5. Permissions / ownership ────────────────────
echo ""
echo "── Permissions ──────────────────────────────"
if [ -d /ssd ]; then
    stat_out=$(stat -c "%U:%G %a" /ssd)
    ok "/ssd owner/perms: $stat_out"
    if [ -w /ssd ]; then
        ok "/ssd is writable by current user ($(whoami))"
    else
        fail "/ssd is NOT writable — check ownership"
    fi
else
    fail "/ssd directory does not exist"
fi

# ── 6. Expected subdirs ───────────────────────────
echo ""
echo "── Expected directories ─────────────────────"
for d in data models .cache/pip .cache/uv .cache/torch .cache/huggingface tmp; do
    if [ -d "/ssd/$d" ]; then
        ok "/ssd/$d exists"
    else
        warn "/ssd/$d missing (run setup_ssd.sh to create)"
    fi
done

# ── 7. Environment variables ──────────────────────
echo ""
echo "── Environment variables ────────────────────"
check_env() {
    local var=$1 expected=$2
    val="${!var}"
    if [ -z "$val" ]; then
        warn "$var not set (default will use home dir, not /ssd)"
    elif [[ "$val" == /ssd* ]]; then
        ok "$var=$val"
    else
        warn "$var=$val (not pointing to /ssd)"
    fi
}
check_env UV_CACHE_DIR        /ssd/.cache/uv
check_env PIP_CACHE_DIR       /ssd/.cache/pip
check_env TORCH_HOME          /ssd/.cache/torch
check_env HF_HOME             /ssd/.cache/huggingface
check_env TRANSFORMERS_CACHE  /ssd/.cache/huggingface/hub
check_env JUPYTER_DATA_DIR    /ssd/.jupyter
check_env TMPDIR              /ssd/tmp

# Check profile.d
if [ -f /etc/profile.d/ssd-caches.sh ]; then
    ok "/etc/profile.d/ssd-caches.sh present"
else
    warn "/etc/profile.d/ssd-caches.sh missing (run setup_ssd.sh)"
fi

# ── 8. pip / uv config ────────────────────────────
echo ""
echo "── Tool config files ────────────────────────"
pip_conf_cache=$(pip config get global.cache-dir 2>/dev/null | tr -d '[:space:]')
if [[ "$pip_conf_cache" == /ssd* ]]; then
    ok "pip.conf cache-dir = $pip_conf_cache"
else
    warn "pip cache not configured to /ssd (got: '${pip_conf_cache:-unset}')"
fi

uv_conf="${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml"
if [ -f "$uv_conf" ] && grep -q "/ssd" "$uv_conf" 2>/dev/null; then
    ok "uv.toml points to /ssd ($uv_conf)"
else
    warn "uv.toml missing or not pointing to /ssd (checked: $uv_conf)"
fi

# ── 9. Conda ──────────────────────────────────────
echo ""
echo "── Conda ────────────────────────────────────"
if command -v conda &>/dev/null; then
    conda_root=$(conda info --base 2>/dev/null)
    if [[ "$conda_root" == /ssd* ]]; then
        ok "conda base env on /ssd: $conda_root"
    else
        warn "conda base env is on root fs: $conda_root (consider moving envs to /ssd)"
    fi
    condarc_pkgs=$(conda config --show pkgs_dirs 2>/dev/null | grep -v "^pkgs_dirs" | head -1 | tr -d ' -')
    if [[ "$condarc_pkgs" == /ssd* ]]; then
        ok "conda pkgs_dirs on /ssd"
    else
        warn "conda pkgs_dirs: $condarc_pkgs (not on /ssd)"
    fi
else
    ok "conda not installed (nothing to check)"
fi

# ── 10. Disk space summary ────────────────────────
echo ""
echo "── Disk space ───────────────────────────────"
df -h / /ssd 2>/dev/null | column -t
echo ""
echo "── Cache sizes on /ssd ──────────────────────"
du -sh /ssd/.cache/* 2>/dev/null | sort -rh || echo "  (no caches yet)"

# ── Summary ───────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Result: $PASS passed  $WARN warnings  $FAIL failed"
echo "══════════════════════════════════════════════"
echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "  Run setup_ssd.sh to fix failures."
elif [ "$WARN" -gt 0 ]; then
    echo "  Run setup_ssd.sh to resolve warnings."
else
    echo "  /ssd is fully configured."
fi
echo ""
