#!/usr/bin/env bash
# =============================================================================
#  proxima-setup.sh — build, install, update and run a Proxima testnet node,
#                     wallet and miner from the master branch of
#                     https://github.com/lunfardo314/proxima
#
#  Targets the current public testnet ("testnet5", node v0.13.x-testnet).
#  A testnet reset keeps the version number but starts a NEW ledger (new genesis),
#  so the script compares the genesis of the public nodes with that of your node DB
#  and re-syncs from a fresh snapshot when they differ (see 'reset').
#
#  Binaries:
#    proxima  — the node (access node; syncs from a snapshot automatically)
#    proxi    — the CLI wallet AND the miner  ('proxi node mine')
#
#  Commands:
#    install       full first-time setup: deps, Go, clone, build, install
#                  binaries, node config, wallet, systemd services, start
#    update        pull master, rebuild, swap binaries, restart services
#                  (keys and configs are kept; the node DB is wiped when it
#                  belongs to an older ledger, or with --reset-db)
#    reset         the network was reset (new genesis): stop, wipe the node DB if
#                  it is from an old ledger, refresh peer IDs, restart, re-sync.
#                  No rebuild; use 'update' when the version changed too
#    build         only clone/pull + compile (no install, no services)
#    start|stop|restart [node|miner|all]
#    status        versions, services, sync state, wallet balance
#    logs [node|miner]   follow the logs
#    remote <update|install|status|reset> user@host [user@host ...]
#                  copy this script to each host over ssh and run it there
#    uninstall     stop + remove services and binaries (keeps data dir)
#
#  Options (flags or environment variables):
#    --home DIR            PROXIMA_HOME   base directory      (~/proxima)
#    --branch NAME         BRANCH         git branch          (master)
#    --bin-dir DIR         BIN_DIR        where binaries go   (/usr/local/bin)
#    --workers N           MINER_WORKERS  miner threads       (all CPUs)
#    --max-khs X           MINER_MAX_KHS  hashrate cap, KH/s  (0 = none)
#    --miner-api URL       MINER_API      node API the miner uses
#                                         (http://127.0.0.1:<api port>)
#    --seeker              WITH_SEEKER=1  also build the Rust nonce_seeker
#                                         (~2x faster per core) and use it
#    --encrypt             ENCRYPT_KEY=1  encrypt the wallet key; passphrase
#                                         is stored root-only for systemd
#    --no-miner            NO_MINER=1     node + wallet only
#    --no-node             NO_NODE=1      wallet + miner only (uses a public
#                                         node API unless --miner-api given)
#    --no-services         NO_SERVICES=1  do not create/start systemd units
#    --reset-db            RESET_DB=1     wipe the node DB on update/reset even if
#                                         the ledger looks unchanged
#    -y | --yes            ASSUME_YES=1   no confirmation prompts
# =============================================================================
set -Eeuo pipefail

SCRIPT_VERSION="2.0.0"
REPO_URL="${REPO_URL:-https://github.com/lunfardo314/proxima.git}"

# ---------- defaults ---------------------------------------------------------
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
PROXIMA_HOME="${PROXIMA_HOME:-$RUN_HOME/proxima}"
BRANCH="${BRANCH:-master}"
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
MINER_WORKERS="${MINER_WORKERS:-}"
MINER_MAX_KHS="${MINER_MAX_KHS:-0}"
MINER_API="${MINER_API:-}"
WITH_SEEKER="${WITH_SEEKER:-0}"
ENCRYPT_KEY="${ENCRYPT_KEY:-0}"
NO_MINER="${NO_MINER:-0}"
NO_NODE="${NO_NODE:-0}"
NO_SERVICES="${NO_SERVICES:-0}"
RESET_DB="${RESET_DB:-0}"
ASSUME_YES="${ASSUME_YES:-0}"

# public testnet access points (also rendered into proxima.yaml by proxi)
PUBLIC_APIS=("http://65.21.170.230:8001" "http://79.137.70.25:8001" "http://51.254.47.76:8001")

NODE_SVC="proxima-node"
MINER_SVC="proxima-miner"
ENV_DIR="/etc/proxima"

# ---------- output helpers ---------------------------------------------------
if [[ -t 1 ]]; then C_B=$'\e[1m'; C_G=$'\e[32m'; C_Y=$'\e[33m'; C_R=$'\e[31m'; C_C=$'\e[36m'; C_0=$'\e[0m'
else C_B=; C_G=; C_Y=; C_R=; C_C=; C_0=; fi
log()  { printf '%s[%s]%s %s\n' "$C_C" "$(date +%H:%M:%S)" "$C_0" "$*"; }
ok()   { printf '%s  ✔%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s  !%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '%s  ✘ %s%s\n' "$C_R" "$*" "$C_0" >&2; exit 1; }
step() { printf '\n%s==> %s%s\n' "$C_B" "$*" "$C_0"; }
trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

confirm() {
  [[ "$ASSUME_YES" == 1 ]] && return 0
  [[ -t 0 ]] || return 0
  read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]
}

# run as root (directly or through sudo)
as_root() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi; }
# run as the service user (drops root when invoked via sudo)
as_user() {
  if [[ $EUID -eq 0 && "$RUN_USER" != root ]]; then sudo -u "$RUN_USER" -H env "PATH=$PATH" "$@"
  else "$@"; fi
}

have_systemd() { [[ "$NO_SERVICES" != 1 ]] && command -v systemctl >/dev/null && [[ -d /run/systemd/system ]]; }

# ---------- paths ------------------------------------------------------------
set_paths() {
  SRC_DIR="$PROXIMA_HOME/src/proxima"
  NODE_DIR="$PROXIMA_HOME/node"
  WALLET_DIR="$PROXIMA_HOME/wallet"
  GO_ROOT="$PROXIMA_HOME/.toolchain/go"
  GO_PATH="$PROXIMA_HOME/.toolchain/gopath"
  BUILD_OUT="$PROXIMA_HOME/build"
  BACKUP_DIR="$PROXIMA_HOME/backups"
  CARGO_HOME_DIR="$PROXIMA_HOME/.toolchain/cargo"
  RUSTUP_HOME_DIR="$PROXIMA_HOME/.toolchain/rustup"
  LEDGER_FILE="$PROXIMA_HOME/.ledger_id"   # "<ledger hash> <genesis unix time> <description>" of the DB
}

# ---------- argument parsing -------------------------------------------------
CMD="${1:-help}"; shift || true
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --home)        PROXIMA_HOME="$2"; shift 2;;
    --branch)      BRANCH="$2"; shift 2;;
    --bin-dir)     BIN_DIR="$2"; shift 2;;
    --workers)     MINER_WORKERS="$2"; shift 2;;
    --max-khs)     MINER_MAX_KHS="$2"; shift 2;;
    --miner-api)   MINER_API="$2"; shift 2;;
    --seeker)      WITH_SEEKER=1; shift;;
    --encrypt)     ENCRYPT_KEY=1; shift;;
    --no-miner)    NO_MINER=1; shift;;
    --no-node)     NO_NODE=1; shift;;
    --no-services) NO_SERVICES=1; shift;;
    --reset-db)    RESET_DB=1; shift;;
    -y|--yes)      ASSUME_YES=1; shift;;
    -h|--help)     CMD=help; shift;;
    *)             ARGS+=("$1"); shift;;
  esac
done
set_paths

# =============================================================================
#  Which ledger is the network on?  (testnet resets keep the version number)
# =============================================================================
# A reset creates a new genesis. The ledger lives in the node DB, not in the binary, so an
# old DB can never sync to the new network. We compare genesis_time_unix of the public nodes
# with the one recorded for (or served by) the local node. The ledger *hash* is deliberately
# not used: ordinary ledger upgrades change it too.
net_get() { # net_get <api path> : first public node that answers
  local u r
  for u in "${PUBLIC_APIS[@]}"; do
    r=$(curl -fs -m 8 "$u$1" 2>/dev/null) && [[ -n "$r" ]] && { printf '%s' "$r"; return 0; }
  done
  return 1
}

NET_HASH=""; NET_GENESIS=""; NET_DESC=""; NET_MINER_VER=""; NET_NODE_VER=""
load_net_info() {
  command -v jq >/dev/null || return 1
  local lc ni
  lc=$(net_get /api/v1/ledger_constants) || return 1
  NET_HASH=$(jq -r '.hash // empty' <<<"$lc" 2>/dev/null || true)
  NET_GENESIS=$(jq -r '.genesis_time_unix // empty' <<<"$lc" 2>/dev/null || true)
  NET_DESC=$(jq -r '.description // empty' <<<"$lc" 2>/dev/null || true)
  NET_MINER_VER=$(jq -r '.miner_version // empty' <<<"$lc" 2>/dev/null || true)
  ni=$(net_get /api/v1/node_info) || ni=""
  NET_NODE_VER=$(jq -r '.version // empty' <<<"$ni" 2>/dev/null || true)
  [[ -n "$NET_GENESIS" ]]
}

local_api() { echo "http://127.0.0.1:$(api_port)"; }

# prints: same | other | none | unknown   (is the node DB from the ledger the network runs now?)
db_ledger_state() {
  [[ -n "$NET_GENESIS" ]] || { echo unknown; return; }
  # a DB that does not exist yet cannot be stale
  if [[ ! -e "$NODE_DIR/proximadb" && ! -e "$NODE_DIR/proximadb.txstore" ]]; then echo none; return; fi
  # 1. the running node tells us itself
  local lg
  lg=$(curl -fs -m 4 "$(local_api)/api/v1/ledger_constants" 2>/dev/null | jq -r '.genesis_time_unix // empty' 2>/dev/null || true)
  if [[ -n "$lg" ]]; then [[ "$lg" == "$NET_GENESIS" ]] && echo same || echo other; return; fi
  # 2. the record written when this script last set the DB up
  if [[ -f "$LEDGER_FILE" ]]; then
    lg=$(awk '{print $2;exit}' "$LEDGER_FILE")
    [[ "$lg" == "$NET_GENESIS" ]] && echo same || echo other; return
  fi
  # 3. no record: a DB last written before the genesis cannot belong to this ledger
  local newest; newest=$(find "$NODE_DIR/proximadb" "$NODE_DIR/proximadb.txstore" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 || true)
  newest=${newest%.*}
  if [[ -n "$newest" ]]; then (( newest < NET_GENESIS )) && echo other || echo unknown; else echo unknown; fi
}

record_ledger() {
  [[ -n "$NET_GENESIS" ]] || return 0
  printf '%s %s %s\n' "$NET_HASH" "$NET_GENESIS" "$NET_DESC" | as_user tee "$LEDGER_FILE" >/dev/null
}

src_miner_version() { # miner version this source tree writes into transits
  grep -hoE 'MinerVersion +uint32 += +[0-9]+' "$SRC_DIR/proxi/node_cmd/mine/mine.go" 2>/dev/null | grep -oE '[0-9]+$' | head -1 || true
}
minor_of() { echo "$1" | sed -E 's/^v?([0-9]+\.[0-9]+).*/\1/'; }

# warn when what we are about to build cannot talk to the network
check_compat() {
  load_net_info || { warn "could not read ledger info from the public nodes (offline?) — skipping compatibility checks"; return 0; }
  log "network ledger: ${C_B}${NET_DESC:-?}${C_0}  genesis $(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS") UTC  nodes ${NET_NODE_VER:-?}"
  if [[ -n "${SRC_VERSION:-}" && -n "$NET_NODE_VER" && "$(minor_of "$SRC_VERSION")" != "$(minor_of "$NET_NODE_VER")" ]]; then
    warn "source is $SRC_VERSION but the network runs $NET_NODE_VER — a different minor version is usually incompatible."
    warn "pick the matching tag/branch with --branch <name> if master is ahead of the network."
  fi
  local mv; mv=$(src_miner_version)
  if [[ -n "$mv" && -n "$NET_MINER_VER" && "$mv" != "$NET_MINER_VER" ]]; then
    warn "miner version in source is $mv, the ledger requires $NET_MINER_VER — mined transits would be invalid."
  fi
}

wipe_node_db() {
  as_user rm -rf "$NODE_DIR/proximadb" "$NODE_DIR/proximadb.txstore" "$NODE_DIR/proximadb.txlog"
  as_user bash -c 'rm -f "$1"/*.snapshot' _ "$NODE_DIR"
  ok "node DB wiped — will restore from the network's latest snapshot"
}

# decide_db_wipe: sets DB_WIPE=1 when the node DB must go. Call BEFORE stopping services.
DB_WIPE=0; DB_STATE=unknown
decide_db_wipe() {
  DB_WIPE=0; DB_STATE=unknown
  [[ "$NO_NODE" == 1 ]] && return 0
  DB_STATE=$(db_ledger_state)
  if [[ "$RESET_DB" == 1 ]]; then DB_WIPE=1
  elif [[ "$DB_STATE" == other ]]; then
    warn "the node DB belongs to an OLD ledger (the network was reset): it cannot sync any more"
    if confirm "Delete node DB (proximadb, proximadb.txstore) and re-sync from a fresh snapshot?"; then DB_WIPE=1; fi
  fi
  return 0
}
remember_ledger() { load_net_info >/dev/null 2>&1 || true; record_ledger; }

# =============================================================================
#  Step 1: system dependencies
# =============================================================================
install_deps() {
  step "System dependencies"
  local need=()
  for c in git curl tar jq; do command -v "$c" >/dev/null || need+=("$c"); done
  # Go binaries are built with CGO_ENABLED=0 (static); a C linker is only needed for Rust
  if [[ "$WITH_SEEKER" == 1 ]]; then command -v cc >/dev/null || need+=(gcc); fi
  if [[ ${#need[@]} -eq 0 ]]; then ok "git, curl, tar, jq present"; return; fi
  log "installing: ${need[*]}"
  if   command -v apt-get >/dev/null; then
    as_root env DEBIAN_FRONTEND=noninteractive apt-get update -qq
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}" ca-certificates
  elif command -v dnf >/dev/null; then as_root dnf install -y "${need[@]}"
  elif command -v yum >/dev/null; then as_root yum install -y "${need[@]}"
  elif command -v pacman >/dev/null; then as_root pacman -Sy --noconfirm "${need[@]}"
  else die "install manually: ${need[*]}"; fi
  ok "dependencies installed"
}

check_clock() {
  command -v timedatectl >/dev/null || return 0
  if timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes; then
    ok "system clock is NTP-synchronized"
  else
    warn "system clock is NOT NTP-synchronized. Proxima ledger time is real time;"
    warn "an unsynced clock makes the node/miner fall behind. Enabling NTP..."
    as_root timedatectl set-ntp true 2>/dev/null || warn "could not enable NTP — run: sudo timedatectl set-ntp true"
  fi
}

# =============================================================================
#  Step 2: source code (clone or fast-forward master)
# =============================================================================
fetch_source() {
  step "Source: $REPO_URL ($BRANCH)"
  as_user mkdir -p "$(dirname "$SRC_DIR")"
  if [[ -d "$SRC_DIR/.git" ]]; then
    local before; before=$(as_user git -C "$SRC_DIR" rev-parse --short HEAD)
    as_user git -C "$SRC_DIR" fetch --quiet --tags origin "$BRANCH"
    as_user git -C "$SRC_DIR" checkout --quiet "$BRANCH"
    as_user git -C "$SRC_DIR" reset --quiet --hard "origin/$BRANCH"
    local after; after=$(as_user git -C "$SRC_DIR" rev-parse --short HEAD)
    [[ "$before" == "$after" ]] && ok "already at $after" || ok "updated $before -> $after"
  else
    as_user git clone --quiet --branch "$BRANCH" "$REPO_URL" "$SRC_DIR"
    ok "cloned at $(as_user git -C "$SRC_DIR" rev-parse --short HEAD)"
  fi
  SRC_VERSION=$(grep -oE 'Version\s+=\s+"[^"]+"' "$SRC_DIR/global/version.go" | cut -d'"' -f2)
  log "source version: ${C_B}$SRC_VERSION${C_0}  ($(as_user git -C "$SRC_DIR" log -1 --format='%h %cd' --date=short))"
}

# =============================================================================
#  Step 3: Go toolchain (exact version from go.mod 'toolchain', private copy)
# =============================================================================
ensure_go() {
  step "Go toolchain"
  local want
  want=$(awk '/^toolchain /{sub("go","",$2);print $2}' "$SRC_DIR/go.mod")
  [[ -n "$want" ]] || want=$(awk '/^go /{print $2}' "$SRC_DIR/go.mod")
  local arch; case "$(uname -m)" in
    x86_64|amd64) arch=amd64;; aarch64|arm64) arch=arm64;; armv6l|armv7l) arch=armv6l;;
    *) die "unsupported CPU: $(uname -m)";; esac
  if [[ -x "$GO_ROOT/bin/go" ]] && "$GO_ROOT/bin/go" version | grep -q "go$want "; then
    ok "go $want (cached)"
  else
    log "downloading go $want for linux/$arch"
    local tmp; tmp=$(as_user mktemp -d -p "$PROXIMA_HOME")
    as_user curl -fsSL "https://go.dev/dl/go$want.linux-$arch.tar.gz" -o "$tmp/go.tgz"
    as_user rm -rf "$GO_ROOT"; as_user mkdir -p "$(dirname "$GO_ROOT")"
    as_user tar -C "$(dirname "$GO_ROOT")" -xzf "$tmp/go.tgz"
    as_user rm -rf "$tmp"
    ok "$("$GO_ROOT/bin/go" version)"
  fi
  GO_ENV=(PATH="$GO_ROOT/bin:$PATH" GOROOT="$GO_ROOT" GOPATH="$GO_PATH"
          GOCACHE="$PROXIMA_HOME/.toolchain/gocache" GOTMPDIR="$PROXIMA_HOME/.toolchain/tmp"
          GOTOOLCHAIN=local GOFLAGS=-buildvcs=true CGO_ENABLED=0)
  as_user mkdir -p "$PROXIMA_HOME/.toolchain/tmp"
}
go_run() { as_user env "${GO_ENV[@]}" go "$@"; }

# =============================================================================
#  Step 4: compile proxima (node) + proxi (wallet & miner) [+ nonce_seeker]
# =============================================================================
low_mem() { # < 3 GB RAM: the go compiler can be OOM-killed with full parallelism
  local kb; kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  (( ${kb:-0} > 0 && ${kb:-0} < 3000000 ))
}

build_binaries() {
  step "Compiling $SRC_VERSION"
  as_user mkdir -p "$BUILD_OUT"
  local par=()
  if low_mem; then par=(-p 1); warn "less than 3 GB RAM: building with -p 1 (slower, avoids OOM)"; fi
  ( cd "$SRC_DIR"
    log "proxima (node)…  (first build downloads modules, a few minutes)"
    go_run build ${par[@]+"${par[@]}"} -trimpath -o "$BUILD_OUT/proxima.new" .
    log "proxi (wallet + miner)…"
    go_run build ${par[@]+"${par[@]}"} -trimpath -o "$BUILD_OUT/proxi.new" ./proxi
  )
  local v1 v2
  v1=$("$BUILD_OUT/proxima.new" version | awk '/Version:/{print $2}')
  v2=$("$BUILD_OUT/proxi.new" version 2>/dev/null | awk '/Version:/{print $2}')
  [[ "$v1" == "$SRC_VERSION" && "$v2" == "$SRC_VERSION" ]] || die "built binaries report '$v1'/'$v2', expected $SRC_VERSION"
  ok "proxima $v1, proxi $v2 (commit $("$BUILD_OUT/proxima.new" version | awk '/hash:/{print substr($3,1,10)}'))"

  if [[ "$WITH_SEEKER" == 1 ]]; then build_seeker; fi
}

build_seeker() {
  log "nonce_seeker (Rust)…"
  export CARGO_HOME="$CARGO_HOME_DIR" RUSTUP_HOME="$RUSTUP_HOME_DIR"
  export PATH="$CARGO_HOME/bin:$PATH"
  if ! command -v cargo >/dev/null; then
    log "installing Rust toolchain (rustup, private to $PROXIMA_HOME)"
    as_user env CARGO_HOME="$CARGO_HOME" RUSTUP_HOME="$RUSTUP_HOME_DIR" \
      sh -c 'curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/dev/null'
  fi
  ( cd "$SRC_DIR/proxi/node_cmd/mine/nonce_seeker"
    as_user env PATH="$PATH" CARGO_HOME="$CARGO_HOME" RUSTUP_HOME="$RUSTUP_HOME_DIR" RUSTFLAGS="-C target-cpu=native" \
      cargo build --release --quiet )
  as_user cp "$SRC_DIR/proxi/node_cmd/mine/nonce_seeker/target/release/nonce_seeker" "$BUILD_OUT/nonce_seeker.new"
  ok "nonce_seeker built"
}

# =============================================================================
#  Step 5: install binaries (atomically; previous ones kept as *.prev)
# =============================================================================
install_binaries() {
  step "Installing binaries to $BIN_DIR"
  as_root mkdir -p "$BIN_DIR"
  local b
  for b in proxima proxi nonce_seeker; do
    [[ -f "$BUILD_OUT/$b.new" ]] || continue
    [[ -f "$BIN_DIR/$b" ]] && as_root cp -f "$BIN_DIR/$b" "$BIN_DIR/$b.prev"
    as_root install -m 0755 "$BUILD_OUT/$b.new" "$BIN_DIR/$b.tmp"
    as_root mv -f "$BIN_DIR/$b.tmp" "$BIN_DIR/$b"
    rm -f "$BUILD_OUT/$b.new" 2>/dev/null || as_root rm -f "$BUILD_OUT/$b.new"
    ok "$b -> $BIN_DIR/$b"
  done
  # install this script too: used by the miner unit (wait-sync) and for 'proxima-setup update'
  local self; self=$(readlink -f "$0")
  [[ "$self" != "$BIN_DIR/proxima-setup" ]] && as_root install -m 0755 "$self" "$BIN_DIR/proxima-setup"
  ok "proxima-setup -> $BIN_DIR/proxima-setup"
  hash -r
}

installed_version() { [[ -x "$BIN_DIR/proxima" ]] && "$BIN_DIR/proxima" version 2>/dev/null | awk '/Version:/{print $2}' || true; }

# =============================================================================
#  Step 6: node configuration (proxima.yaml)
# =============================================================================
api_port() {
  local p; p=$(awk '/^api:/{f=1;next} f&&/^[^ #]/{f=0} f&&/^  port:/{print $2;exit}' "$NODE_DIR/proxima.yaml" 2>/dev/null || true)
  echo "${p:-8000}"
}
peering_port() {
  local p; p=$(awk '/^  host:/{f=1;next} f&&/^    port:/{print $2;exit}' "$NODE_DIR/proxima.yaml" 2>/dev/null || true)
  echo "${p:-4000}"
}

# The static peers in proxima.yaml carry the libp2p host ID of each public node. A node whose
# host key was regenerated (typical for a testnet reset) gets a new ID, and the old entry then
# silently fails to connect. Rewrite the IDs of the public-node entries from /api/v1/node_info.
refresh_public_peers() {
  local yaml="$NODE_DIR/proxima.yaml"
  [[ -f "$yaml" ]] && command -v jq >/dev/null || return 0
  local u ip id old n=0
  for u in "${PUBLIC_APIS[@]}"; do
    ip=${u#http://}; ip=${ip%%:*}
    id=$(curl -fs -m 8 "$u/api/v1/node_info" 2>/dev/null | jq -r '.id // empty' 2>/dev/null || true)
    [[ "$id" =~ ^12D3KooW[A-Za-z0-9]+$ ]] || continue
    old=$(grep -oE "/ip4/$ip/udp/[0-9]+/quic-v1/p2p/[A-Za-z0-9]+" "$yaml" 2>/dev/null | head -1 | awk -F/p2p/ '{print $2}' || true)
    [[ -n "$old" && "$old" != "$id" ]] || continue
    as_user sed -i -E "s#(/ip4/$ip/udp/[0-9]+/quic-v1/p2p/)[A-Za-z0-9]+#\1$id#" "$yaml"
    n=$((n+1)); log "peer $ip: host ID $old -> $id"
  done
  if (( n > 0 )); then ok "refreshed $n public peer ID(s) in proxima.yaml"; fi
  return 0
}

configure_node() {
  [[ "$NO_NODE" == 1 ]] && return
  step "Node configuration ($NODE_DIR)"
  as_user mkdir -p "$NODE_DIR"
  if [[ -f "$NODE_DIR/proxima.yaml" ]]; then
    ok "keeping existing proxima.yaml (host ID $(awk '/^    id:/{print $2;exit}' "$NODE_DIR/proxima.yaml"))"
    refresh_public_peers
  else
    ( cd "$NODE_DIR" && as_user "$BIN_DIR/proxi" config node -v >/dev/null )
    as_user chmod 600 "$NODE_DIR/proxima.yaml"
    ok "proxima.yaml created: public testnet peers + snapshot/sync sources pre-filled"
  fi
  # firewall: peering is QUIC over UDP
  local pport; pport=$(peering_port)
  if command -v ufw >/dev/null && as_root ufw status 2>/dev/null | grep -q "Status: active"; then
    as_root ufw allow "$pport/udp" >/dev/null && ok "ufw: opened $pport/udp for peering"
  else
    log "make sure UDP port $pport (peering, QUIC) is reachable from the internet"
  fi
}

# =============================================================================
#  Step 7: wallet (proxi.yaml + proxima.key)
# =============================================================================
key_is_encrypted() { grep -q '"crypto"' "$WALLET_DIR/proxima.key" 2>/dev/null; }

# Newer proxi reads 'api.node_url' (old name: 'api.endpoint') and a witness list 'api.node_urls':
# the wallet only accepts the node's ledger library together with a branch proof confirmed by these
# witnesses. Bring a profile written by an older proxi up to date (keeps everything else).
migrate_wallet_profile() {
  local y="$WALLET_DIR/proxi.yaml"
  [[ -f "$y" ]] || return 0
  if grep -qE '^\s+endpoint:' "$y" && ! grep -qE '^\s+node_url:' "$y"; then
    as_user sed -i -E 's#^(\s+)endpoint:#\1node_url:#' "$y"
    ok "wallet profile: api.endpoint renamed to api.node_url"
  fi
  if ! grep -qE '^\s+node_urls:' "$y"; then
    local ind line=""
    ind=$(grep -E '^\s+node_url:' "$y" | head -1 | sed -E 's/^(\s+).*/\1/' || true)
    if [[ -n "$ind" ]]; then
      local u; for u in "${PUBLIC_APIS[@]}"; do line+="${ind}    - $u\n"; done
      as_user sed -i -E "/^\s+node_url:/a\\${ind}node_urls:\n${line%\\n}" "$y"
      ok "wallet profile: added api.node_urls (witness nodes for the library proof)"
    fi
  fi
  return 0
}

configure_wallet() {
  step "Wallet ($WALLET_DIR)"
  as_user mkdir -p "$WALLET_DIR"; as_user chmod 700 "$WALLET_DIR"
  if [[ -f "$WALLET_DIR/proxi.yaml" && -f "$WALLET_DIR/proxima.key" ]]; then
    ok "keeping existing wallet, holder $(awk '/holder_id:/{print $2;exit}' "$WALLET_DIR/proxi.yaml")"
    migrate_wallet_profile
  else
    # existing key -> answer 'y' to "use it?"; new key -> 'n' to "encrypt?" (that needs a TTY,
    # encryption is done afterwards with 'proxi util key encrypt', see --encrypt)
    local answer=n; [[ -f "$WALLET_DIR/proxima.key" ]] && answer=y
    ( cd "$WALLET_DIR" && printf '%s\n' "$answer" | as_user "$BIN_DIR/proxi" config wallet -v >/dev/null )
    ok "new wallet, holder $(awk '/holder_id:/{print $2;exit}' "$WALLET_DIR/proxi.yaml")"
    backup_key
  fi

  # point the wallet at our own node (or a public one when there is no node)
  local url="${MINER_API:-}"
  if [[ -z "$url" ]]; then
    if [[ "$NO_NODE" == 1 ]]; then url="${PUBLIC_APIS[0]}"; else url="http://127.0.0.1:$(api_port)"; fi
  fi
  as_user sed -i -E "s#^(\s*node_url:).*#\1 $url#" "$WALLET_DIR/proxi.yaml"
  MINER_API="$url"
  ok "wallet/miner node API: $url"

  if [[ "$ENCRYPT_KEY" == 1 ]] && ! key_is_encrypted; then
    [[ -t 0 ]] || die "--encrypt needs an interactive terminal"
    log "encrypting proxima.key (min. passphrase length enforced by proxi)"
    ( cd "$WALLET_DIR" && as_user "$BIN_DIR/proxi" util key encrypt --file proxima.key )
    store_passphrase
  fi
  if key_is_encrypted && [[ ! -f "$ENV_DIR/miner.env" ]] && [[ "$NO_MINER" != 1 ]]; then
    warn "key is encrypted: the miner service needs the passphrase"; store_passphrase
  fi
}

store_passphrase() {
  have_systemd || return 0
  local p; read -r -s -p "Passphrase for the miner service (stored root-only in $ENV_DIR/miner.env): " p; echo
  [[ -n "$p" ]] || die "empty passphrase"
  as_root mkdir -p "$ENV_DIR"; as_root chmod 700 "$ENV_DIR"
  printf 'PROXIMA_KEY_PASSPHRASE=%s\n' "$p" | as_root tee "$ENV_DIR/miner.env" >/dev/null
  as_root chmod 600 "$ENV_DIR/miner.env"
  ok "passphrase stored"
}

backup_key() {
  as_user mkdir -p "$BACKUP_DIR"; as_user chmod 700 "$BACKUP_DIR"
  local f="$BACKUP_DIR/wallet-$(date +%Y%m%d-%H%M%S).tar.gz"
  as_user tar -C "$WALLET_DIR" -czf "$f" proxima.key proxi.yaml
  as_user chmod 600 "$f"
  ok "wallet backup: $f  ${C_Y}(copy it somewhere safe!)${C_0}"
}

# =============================================================================
#  Step 8: systemd services
# =============================================================================
write_units() {
  have_systemd || { warn "systemd not available (or --no-services): skipping services"; return; }
  step "systemd services"
  local workers_flag="" khs_flag="" seeker_flag="" stream_flag
  [[ -n "$MINER_WORKERS" ]] && workers_flag="--workers $MINER_WORKERS"
  [[ "$MINER_MAX_KHS" != 0 ]] && khs_flag="--max-hashrate-khs $MINER_MAX_KHS"
  [[ "$WITH_SEEKER" == 1 || -x "$BIN_DIR/nonce_seeker" ]] && seeker_flag="--seeker"
  # subscribe to several independent mining streams (no single node can withhold)
  stream_flag="--stream $(IFS=,; echo "${PUBLIC_APIS[*]}")"

  if [[ "$NO_NODE" != 1 ]]; then
    as_root tee "/etc/systemd/system/$NODE_SVC.service" >/dev/null <<EOF
[Unit]
Description=Proxima node (access node)
After=network-online.target time-sync.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$NODE_DIR
ExecStart=$BIN_DIR/proxima
Restart=on-failure
RestartSec=5
TimeoutStopSec=120
KillSignal=SIGTERM
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF
    ok "$NODE_SVC.service"
  fi

  if [[ "$NO_MINER" != 1 ]]; then
    local after="network-online.target"; local wants="network-online.target"
    [[ "$NO_NODE" != 1 ]] && { after+=" $NODE_SVC.service"; wants+=" $NODE_SVC.service"; }
    as_root tee "/etc/systemd/system/$MINER_SVC.service" >/dev/null <<EOF
[Unit]
Description=Proxima miner (proxi node mine)
After=$after
Wants=$wants

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$WALLET_DIR
Environment=PATH=$BIN_DIR:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=-$ENV_DIR/miner.env
ExecStartPre=$BIN_DIR/proxima-setup wait-sync $MINER_API
ExecStart=$BIN_DIR/proxi node mine $workers_flag $khs_flag $seeker_flag $stream_flag
Restart=always
RestartSec=15
TimeoutStartSec=infinity
Nice=10

[Install]
WantedBy=multi-user.target
EOF
    ok "$MINER_SVC.service ($MINER_API${seeker_flag:+, nonce_seeker})"
  fi
  as_root systemctl daemon-reload
}

svc_list() {
  local which="${1:-all}" l=()
  [[ "$which" =~ ^(all|node)$  && "$NO_NODE"  != 1 && -f /etc/systemd/system/$NODE_SVC.service  ]] && l+=("$NODE_SVC")
  [[ "$which" =~ ^(all|miner)$ && "$NO_MINER" != 1 && -f /etc/systemd/system/$MINER_SVC.service ]] && l+=("$MINER_SVC")
  echo "${l[@]:-}"
}

svc() { # svc <start|stop|restart|enable> [node|miner|all]
  local action="$1" s; local list; list=$(svc_list "${2:-all}")
  [[ "$action" == stop ]] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')   # miner first
  if have_systemd && [[ -n "$list" ]]; then
    for s in $list; do
      # --no-block: the miner unit's ExecStartPre waits for node sync, never block on it
      if [[ "$action" == enable ]]; then as_root systemctl enable "$s" >/dev/null 2>&1
        as_root systemctl start --no-block "$s"
      elif [[ "$action" == stop ]]; then as_root systemctl stop "$s"
      else as_root systemctl "$action" --no-block "$s"; fi
      ok "$action $s"
    done
  else
    fallback_svc "$action" "${2:-all}"
  fi
}

# ---- no-systemd fallback: background processes with pid files ---------------
# spawn_detached <workdir> <logfile> <pidfile> <cmd...>: fully detached (own session,
# no inherited stdio) so that callers piping our output (ssh, | tee) are not held open
spawn_detached() {
  local wd="$1" logf="$2" pidf="$3"; shift 3
  local runner=(); [[ $EUID -eq 0 && "$RUN_USER" != root ]] && runner=(sudo -u "$RUN_USER" -H)
  ( cd "$wd" || exit 1
    exec "${runner[@]}" setsid nohup "$@" </dev/null >>"$logf" 2>&1 ) &
  echo $! >"$pidf"
  sleep 1
  # with sudo in between, record the real process
  local real esc; esc=$(printf '%s' "$1" | sed 's/[][\\.^$*+?(){}|]/\\&/g')
  real=$(pgrep -n -f -- "^$esc( |\$)" 2>/dev/null || true)
  if [[ -n "$real" ]]; then echo "$real" >"$pidf"; fi
  return 0
}

fallback_svc() {
  local action="$1" which="$2"
  [[ "$action" == enable ]] && action=start
  if [[ "$action" == stop || "$action" == restart ]]; then
    for n in miner node; do
      [[ "$which" =~ ^(all|$n)$ ]] || continue
      local pf="$PROXIMA_HOME/$n.pid"
      if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then kill -TERM "$(cat "$pf")"; ok "stopped $n"; sleep 3; fi
      rm -f "$pf"
    done
  fi
  [[ "$action" == stop ]] && return
  if [[ "$which" =~ ^(all|node)$ && "$NO_NODE" != 1 ]]; then
    spawn_detached "$NODE_DIR" "$NODE_DIR/stdout.log" "$PROXIMA_HOME/node.pid" "$BIN_DIR/proxima"
    ok "node started in background (pid $(cat "$PROXIMA_HOME/node.pid"), log $NODE_DIR/proxima.log)"
  fi
  if [[ "$which" =~ ^(all|miner)$ && "$NO_MINER" != 1 ]]; then
    local extra=(); [[ -n "$MINER_WORKERS" ]] && extra+=(--workers "$MINER_WORKERS")
    [[ "$MINER_MAX_KHS" != 0 ]] && extra+=(--max-hashrate-khs "$MINER_MAX_KHS")
    [[ -x "$BIN_DIR/nonce_seeker" ]] && extra+=(--seeker)
    spawn_detached "$WALLET_DIR" "$WALLET_DIR/miner.log" "$PROXIMA_HOME/miner.pid" \
      "$BIN_DIR/proxi" node mine "${extra[@]}" --stream "$(IFS=,; echo "${PUBLIC_APIS[*]}")"
    ok "miner started in background (pid $(cat "$PROXIMA_HOME/miner.pid"), log $WALLET_DIR/miner.log)"
  fi
}

# =============================================================================
#  Step 9: wait for sync / status
# =============================================================================
sync_info() { curl -fs -m 5 "$1/api/v1/sync_info" 2>/dev/null || true; }
is_synced() { sync_info "$1" | grep -q '"synced": *true'; }

# used by the miner unit: block until the node API reports synced (max ~1h)
cmd_wait_sync() {
  local url="${1:-http://127.0.0.1:8000}" i
  for i in $(seq 1 360); do is_synced "$url" && { echo "node $url is synced"; return 0; }
    (( i % 6 == 1 )) && echo "waiting for $url to sync…"; sleep 10; done
  echo "node $url not synced after 1h, starting the miner anyway"
}

wait_for_sync() {
  [[ "$NO_NODE" == 1 ]] && return
  step "Waiting for the node to restore the snapshot and sync (Ctrl-C to skip)"
  local url="http://127.0.0.1:$(api_port)" i s
  for i in $(seq 1 180); do
    s=$(sync_info "$url")
    if [[ -n "$s" ]]; then
      if echo "$s" | grep -q '"synced": *true'; then ok "node synced: $(echo "$s" | tr -d '\n ' )"; return; fi
      local cur lrb; cur=$(echo "$s" | jq -r .current_slot 2>/dev/null); lrb=$(echo "$s" | jq -r .lrb_slot 2>/dev/null)
      printf '\r  … syncing: LRB slot %s, current %s (%s behind)      ' "$lrb" "$cur" "$(( ${cur:-0} - ${lrb:-0} ))"
    else printf '\r  … node API not up yet / restoring snapshot (%ds)   ' $((i*10)); fi
    sleep 10
  done
  echo; warn "node not synced after 30 min — check: $0 logs node (the miner waits for it)"
}

cmd_status() {
  step "Network"
  if load_net_info; then
    printf '  ledger    : %s\n  genesis   : %s UTC (hash %s…)\n  nodes run : %s | miner version %s\n' \
      "${NET_DESC:-?}" "$(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS")" "${NET_HASH:0:12}" \
      "${NET_NODE_VER:-?}" "${NET_MINER_VER:-?}"
    if [[ "$NO_NODE" != 1 && -f "$NODE_DIR/proxima.yaml" && "$(db_ledger_state)" == other ]]; then
      warn "this node's DB is from an OLD ledger — run: $0 reset"
    fi
  else
    warn "public nodes not reachable"
  fi
  step "Versions"
  printf '  installed : proxima %s | proxi %s%s\n' "$(installed_version)" \
    "$("$BIN_DIR/proxi" version 2>/dev/null | awk '/Version:/{print $2}')" \
    "$([[ -x $BIN_DIR/nonce_seeker ]] && echo ' | nonce_seeker yes')"
  if [[ -d "$SRC_DIR/.git" ]]; then
    git -C "$SRC_DIR" fetch --quiet origin "$BRANCH" 2>/dev/null || true
    local remote_v; remote_v=$(git -C "$SRC_DIR" show "origin/$BRANCH:global/version.go" 2>/dev/null | grep -oE 'Version\s+=\s+"[^"]+"' | cut -d'"' -f2)
    printf '  upstream  : %s %s (%s)\n' "$BRANCH" "$remote_v" "$(git -C "$SRC_DIR" log -1 --format=%h "origin/$BRANCH" 2>/dev/null)"
    [[ -n "$remote_v" && "$remote_v" != "$(installed_version)" ]] && warn "update available: $0 update"
  fi
  step "Services"
  if have_systemd; then
    for s in $(svc_list all); do printf '  %-15s %s\n' "$s" "$(systemctl is-active "$s" 2>/dev/null || true)"; done
  else
    for n in node miner; do local pf="$PROXIMA_HOME/$n.pid"
      printf '  %-15s %s\n' "$n" "$([[ -f $pf ]] && kill -0 "$(cat "$pf")" 2>/dev/null && echo running || echo stopped)"; done
  fi
  if [[ "$NO_NODE" != 1 && -f "$NODE_DIR/proxima.yaml" ]]; then
    step "Local node"; echo "  $(sync_info "http://127.0.0.1:$(api_port)" | tr -d '\n ' || echo 'API not reachable')"
  fi
  step "Network (public node)"; echo "  $(sync_info "${PUBLIC_APIS[0]}" | tr -d '\n ')"
  if [[ -f "$WALLET_DIR/proxi.yaml" ]]; then
    step "Wallet"
    ( cd "$WALLET_DIR" && timeout 30 "$BIN_DIR/proxi" node balance 2>&1 | grep -v '^Command line' | sed 's/^/  /' ) || true
    ( cd "$WALLET_DIR" && timeout 30 "$BIN_DIR/proxi" node delegate status 2>&1 | grep -v '^Command line' | tail -n 15 | sed 's/^/  /' ) || true
  fi
}

cmd_logs() {
  local which="${1:-miner}"
  [[ "$which" =~ ^(node|miner)$ ]] || die "usage: $0 logs [node|miner]"
  # not 'exec': as_root is a shell function. Ctrl-C ends the follow; that is not an error.
  if have_systemd; then
    local svc_name="$MINER_SVC"; if [[ "$which" == node ]]; then svc_name="$NODE_SVC"; fi
    as_root journalctl -u "$svc_name" -f -n 100 || true
  else
    local logf="$WALLET_DIR/miner.log"; if [[ "$which" == node ]]; then logf="$NODE_DIR/proxima.log"; fi
    tail -F "$logf" || true
  fi
}

# =============================================================================
#  Commands
# =============================================================================
preflight() {
  [[ "$(uname -s)" == Linux ]] || die "Linux only"
  if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    log "sudo is needed for packages, $BIN_DIR and systemd — you may be asked for your password"
    sudo -v || die "sudo required"
  fi
  as_user mkdir -p "$PROXIMA_HOME"
  local free; free=$(df -Pk "$PROXIMA_HOME" | awk 'NR==2{print int($4/1024/1024)}')
  (( free >= 5 )) || die "only ${free} GB free in $PROXIMA_HOME (need >= 5 GB for build + DB)"
  (( free >= 50 )) || warn "only ${free} GB free: the tx store grows over time, plan for more disk"
}

summary() {
  step "Done — Proxima $(installed_version)"
  cat <<EOF
  ${C_B}Layout${C_0}
    source   : $SRC_DIR
    node     : $NODE_DIR   (proxima.yaml, proximadb/, proxima.log)
    wallet   : $WALLET_DIR (proxi.yaml, proxima.key)  <- miner runs here
    backups  : $BACKUP_DIR
    binaries : $BIN_DIR/proxima (node), $BIN_DIR/proxi (wallet + miner)

  ${C_B}Everyday commands${C_0}
    $0 status            versions, sync, balance, delegations
    $0 logs miner|node   follow logs
    $0 update            pull master, rebuild, restart (keeps keys)
    $0 reset             network was reset: wipe the old node DB and re-sync
    cd $WALLET_DIR && proxi node balance
    cd $WALLET_DIR && proxi node delegate status

  ${C_B}Explorers on your node${C_0}: http://<this-host>:$( [[ -f $NODE_DIR/proxima.yaml ]] && api_port || echo 8000)/chain_explorer  /dag_explorer  /peers
EOF
}

cmd_build() { preflight; install_deps; fetch_source; ensure_go; build_binaries; ok "binaries in $BUILD_OUT (*.new)"; }

cmd_install() {
  step "Proxima setup $SCRIPT_VERSION — user=$RUN_USER home=$PROXIMA_HOME branch=$BRANCH"
  preflight; install_deps; check_clock
  fetch_source; check_compat; ensure_go; build_binaries
  decide_db_wipe
  svc stop all 2>/dev/null || true
  [[ "$DB_WIPE" == 1 ]] && wipe_node_db
  install_binaries
  configure_node; configure_wallet
  write_units
  [[ "$NO_NODE" != 1 ]] && remember_ledger
  [[ "$NO_NODE" != 1 ]] && svc enable node
  if [[ "$NO_MINER" != 1 ]]; then
    if have_systemd; then svc enable miner
      log "the miner service waits until the node is synced, then starts mining"
    else wait_for_sync; svc start miner; fi
  fi
  have_systemd && [[ "$NO_NODE" != 1 ]] && wait_for_sync || true
  summary
}

# start everything. With systemd the miner unit waits for the node by itself (ExecStartPre);
# without it the miner would die on a node that is not answering yet, so go step by step.
start_and_sync() {
  if have_systemd; then
    svc start all
    [[ "$NO_NODE" != 1 ]] && wait_for_sync || true
  else
    svc start node
    wait_for_sync || true
    svc start miner
  fi
}

# what is installed here? sets NO_NODE / NO_MINER / WITH_SEEKER / MINER_API accordingly
detect_install() {
  [[ -f "$WALLET_DIR/proxi.yaml" || -f "$NODE_DIR/proxima.yaml" ]] || die "nothing installed in $PROXIMA_HOME — run: $0 install"
  [[ -f "$NODE_DIR/proxima.yaml" ]] || NO_NODE=1
  [[ -f "/etc/systemd/system/$MINER_SVC.service" || -f "$PROXIMA_HOME/miner.pid" || -f "$WALLET_DIR/miner.log" ]] || NO_MINER=1
  [[ -x "$BIN_DIR/nonce_seeker" ]] && WITH_SEEKER=1
  MINER_API=$(awk '/node_url:/{print $2;exit}' "$WALLET_DIR/proxi.yaml" 2>/dev/null || true)
  [[ -n "$MINER_API" ]] || MINER_API=$(awk '/endpoint:/{print $2;exit}' "$WALLET_DIR/proxi.yaml" 2>/dev/null || echo "http://127.0.0.1:8000")
  return 0
}

cmd_update() {
  step "Updating Proxima — branch $BRANCH"
  detect_install
  preflight; install_deps
  local old_v; old_v=$(installed_version)
  fetch_source
  check_compat
  decide_db_wipe
  if [[ "$old_v" == "$SRC_VERSION" && "$DB_WIPE" != 1 && -z "${FORCE_REBUILD:-}" ]]; then
    local inst_hash src_hash
    inst_hash=$("$BIN_DIR/proxima" version | awk '/hash:/{print $3}')
    src_hash=$(git -C "$SRC_DIR" rev-parse HEAD)
    if [[ "$inst_hash" == "$src_hash" ]]; then ok "already running $old_v ($src_hash) — nothing to do"; return; fi
  fi
  ensure_go; build_binaries
  log "upgrade: ${old_v:-none} -> $SRC_VERSION"

  svc stop all || true
  install_binaries
  backup_key 2>/dev/null || true
  [[ "$DB_WIPE" == 1 ]] && wipe_node_db
  migrate_wallet_profile
  [[ "$NO_NODE" != 1 ]] && { refresh_public_peers; remember_ledger; }
  write_units
  start_and_sync
  ok "updated to $SRC_VERSION"
}

# The network was reset (new genesis, same or newer version): no rebuild needed, just a fresh DB.
cmd_reset() {
  step "Re-joining the network after a reset"
  detect_install
  [[ "$NO_NODE" == 1 ]] && die "no node installed here — nothing to reset (the wallet keeps working)"
  preflight
  load_net_info || die "cannot read ledger info from the public nodes — check your connection"
  log "network ledger: ${C_B}${NET_DESC:-?}${C_0}  genesis $(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS") UTC  nodes ${NET_NODE_VER:-?}"
  local inst; inst=$(installed_version)
  if [[ -n "$inst" && -n "$NET_NODE_VER" && "$(minor_of "$inst")" != "$(minor_of "$NET_NODE_VER")" ]]; then
    warn "installed $inst, network runs $NET_NODE_VER — run '$0 update' to get matching binaries"
  fi
  decide_db_wipe
  case "$DB_STATE" in
    same)  ok "node DB already belongs to the current ledger" ;;
    other) : ;;
    *)     log "ledger of the node DB: $DB_STATE" ;;
  esac
  svc stop all || true
  [[ "$DB_WIPE" == 1 ]] && wipe_node_db
  migrate_wallet_profile
  refresh_public_peers
  remember_ledger
  write_units
  start_and_sync
  ok "reset done"
}

cmd_uninstall() {
  confirm "Stop and remove services and binaries? (data in $PROXIMA_HOME is kept)" || exit 0
  svc stop all 2>/dev/null || true
  if have_systemd; then
    for s in "$NODE_SVC" "$MINER_SVC"; do as_root systemctl disable "$s" 2>/dev/null || true
      as_root rm -f "/etc/systemd/system/$s.service"; done
    as_root systemctl daemon-reload
  fi
  for b in proxima proxi nonce_seeker; do as_root rm -f "$BIN_DIR/$b" "$BIN_DIR/$b.prev"; done
  ok "removed. Keys and data remain in $PROXIMA_HOME"
}

cmd_remote() {
  local sub="${ARGS[0]:-}"; [[ "$sub" =~ ^(install|update|status|reset)$ ]] || die "usage: $0 remote <install|update|status|reset> user@host ..."
  [[ ${#ARGS[@]} -ge 2 ]] || die "no hosts given"
  local passthru=(); [[ "$BRANCH" != master ]] && passthru+=(--branch "$BRANCH")
  [[ "$WITH_SEEKER" == 1 ]] && passthru+=(--seeker); [[ "$RESET_DB" == 1 ]] && passthru+=(--reset-db)
  [[ -n "$MINER_WORKERS" ]] && passthru+=(--workers "$MINER_WORKERS")
  [[ "$NO_MINER" == 1 ]] && passthru+=(--no-miner)
  local h
  for h in "${ARGS[@]:1}"; do
    step "[$h] $sub"
    scp -q "$0" "$h:/tmp/proxima-setup.sh"
    ssh -t "$h" "bash /tmp/proxima-setup.sh $sub -y ${passthru[*]:-}" || warn "[$h] failed"
  done
}

usage() { awk 'NR>1 && /^#/{print; next} NR>1{exit}' "$0" | sed 's/^# \{0,1\}//' | grep -v '^====='; }

case "$CMD" in
  install)   cmd_install;;
  update)    cmd_update;;
  reset)     cmd_reset;;
  build)     cmd_build;;
  start|stop|restart) svc "$CMD" "${ARGS[0]:-all}";;
  status)    cmd_status;;
  logs)      cmd_logs "${ARGS[0]:-miner}";;
  remote)    cmd_remote;;
  wait-sync) cmd_wait_sync "${ARGS[0]:-}";;
  uninstall) cmd_uninstall;;
  help|*)    usage;;
esac
