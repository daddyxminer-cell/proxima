#!/usr/bin/env bash
# =============================================================================
#  proxima-setup.sh — build, install, update and run a Proxima testnet node,
#                     wallet and miner from the master branch of
#                     https://github.com/lunfardo314/proxima
#
#  Targets the current public network ("take 1", v1.0.x-take1, miner version 4).
#  v1.0.1-take1 only changes `proxi` (miner fix); the ledger and the nodes are unchanged,
#  so node operators do not need to upgrade, but miners should.
#  A network reset keeps the version number but starts a NEW ledger (new genesis),
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
#    status        colour dashboard: node IP:port + peering address, sync, every
#                  connected peer (RTT, traffic), wallet balance, delegations,
#                  miner hashrate / difficulty / minted / expected rewards
#    peers         all peers with full IDs and every multiaddress
#    watch [secs]  the status dashboard, refreshed in place (default 10 s)
#    logs [node|miner]   follow the logs
#    remote <update|install|status|reset> user@host [user@host ...]
#                  copy this script to each host over ssh and run it there
#    uninstall     stop + remove services and binaries (keeps data dir)
#
#  Options (flags or environment variables):
#    --home DIR            PROXIMA_HOME   base directory      (~/proxima)
#    --branch NAME         BRANCH         git branch/tag/commit (master; e.g. v1.0.1-take1)
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
#    --no-color            NO_COLOR=1     plain output (also automatic when piped)
#    -y | --yes            ASSUME_YES=1   no confirmation prompts
# =============================================================================
set -Eeuo pipefail

SCRIPT_VERSION="2.2.0"
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
PUBLIC_NAMES=(hloc0 oseq1 oloc2)    # same order as PUBLIC_APIS

NODE_SVC="proxima-node"
MINER_SVC="proxima-miner"
ENV_DIR="/etc/proxima"

# ---------- output helpers ---------------------------------------------------
COLOR="${COLOR:-1}"
setup_colors() {
  if [[ -z "${NO_COLOR:-}" && "$COLOR" != 0 && ( -t 1 || -n "${FORCE_COLOR:-}" ) ]]; then
    C_B=$'\e[1m'; C_D=$'\e[90m'; C_G=$'\e[92m'; C_Y=$'\e[93m'; C_R=$'\e[91m'; C_C=$'\e[96m'
    C_M=$'\e[95m'; C_BL=$'\e[94m'; C_W=$'\e[97m'; C_0=$'\e[0m'
  else C_B=; C_D=; C_G=; C_Y=; C_R=; C_C=; C_M=; C_BL=; C_W=; C_0=; fi
}
setup_colors
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
    --no-color)    COLOR=0; shift;;
    --color)       FORCE_COLOR=1; shift;;
    -y|--yes)      ASSUME_YES=1; shift;;
    -h|--help)     CMD=help; shift;;
    *)             ARGS+=("$1"); shift;;
  esac
done
set_paths
setup_colors

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
NET_UNIT=1000000; NET_TICKER=PROX; NET_REWARD=95000000; NET_SLOT_SEC=10.24
load_net_info() {
  command -v jq >/dev/null || return 1
  local lc ni
  lc=$(net_get /api/v1/ledger_constants) || return 1
  NET_HASH=$(jq -r '.hash // empty' <<<"$lc" 2>/dev/null || true)
  NET_GENESIS=$(jq -r '.genesis_time_unix // empty' <<<"$lc" 2>/dev/null || true)
  NET_DESC=$(jq -r '.description // empty' <<<"$lc" 2>/dev/null || true)
  NET_MINER_VER=$(jq -r '.miner_version // empty' <<<"$lc" 2>/dev/null || true)
  NET_UNIT=$(jq -r '.smallest_amounts_per_base_token // empty' <<<"$lc" 2>/dev/null || true); NET_UNIT=${NET_UNIT:-1000000}
  NET_TICKER=$(jq -r '.base_token_name_ticker // empty' <<<"$lc" 2>/dev/null || true); NET_TICKER=${NET_TICKER:-PROX}
  NET_REWARD=$(jq -r '.mine_amount_base // empty' <<<"$lc" 2>/dev/null || true); NET_REWARD=${NET_REWARD:-95000000}
  NET_SLOT_SEC=$(jq -r 'try (.ticks_per_slot * .tick_duration_ns / 1000000000) catch empty' <<<"$lc" 2>/dev/null || true); NET_SLOT_SEC=${NET_SLOT_SEC:-10.24}
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
  if [[ ! -d "$SRC_DIR/.git" ]]; then
    as_user git clone --quiet "$REPO_URL" "$SRC_DIR"
  fi
  local before; before=$(as_user git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo none)
  as_user git -C "$SRC_DIR" fetch --quiet --tags origin
  if as_user git -C "$SRC_DIR" rev-parse --verify --quiet "refs/remotes/origin/$BRANCH" >/dev/null; then
    as_user git -C "$SRC_DIR" checkout --quiet -B "$BRANCH" "origin/$BRANCH"
  else # a tag or a commit: pin the build to exactly that
    as_user git -C "$SRC_DIR" fetch --quiet origin "$BRANCH" 2>/dev/null || true
    as_user git -C "$SRC_DIR" checkout --quiet --detach "$BRANCH" 2>/dev/null \
      || die "'$BRANCH' is not a branch, tag or commit of $REPO_URL"
    log "pinned to ${C_B}$BRANCH${C_0} (tag/commit, not a moving branch)"
  fi
  local after; after=$(as_user git -C "$SRC_DIR" rev-parse --short HEAD)
  if [[ "$before" == "$after" ]]; then ok "already at $after"; else ok "now at $after (was $before)"; fi
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
ExecStart=$BIN_DIR/proxi node mine -v $workers_flag $khs_flag $seeker_flag $stream_flag
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
    local extra=(-v); [[ -n "$MINER_WORKERS" ]] && extra+=(--workers "$MINER_WORKERS")
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

# =============================================================================
#  Dashboard:  status  |  peers  |  watch
# =============================================================================
DASH_W=78

rule()  { local s; printf -v s '%*s' "$1" ''; printf '%s' "${s// /─}"; }
rep()   { local s="" i; for ((i = 0; i < $2; i++)); do s+="$1"; done; printf '%s' "$s"; }
sect()  { printf '\n  %s%s▌ %s%s %s' "$C_B" "$C_M" "$1" "$C_0" "$C_D"; rule $((DASH_W - 6 - ${#1})); printf '%s\n' "$C_0"; }
kv()    { printf '  %s%-17s%s %s\n' "$C_D" "$1" "$C_0" "$2"; }
note()  { printf '  %s%s%s\n' "$C_D" "$*" "$C_0"; }
pc()    { printf '%s%-*s%s' "$1" "$2" "$3" "$C_0"; }     # colour, width, text   (left aligned)
pcr()   { printf '%s%*s%s'  "$1" "$2" "$3" "$C_0"; }     # colour, width, text   (right aligned)
dot_on()  { printf '%s●%s' "$C_G" "$C_0"; }
dot_off() { printf '%s○%s' "$C_R" "$C_0"; }
dot_mid() { printf '%s◐%s' "$C_Y" "$C_0"; }

thou()  { sed -E ':a;s/^([0-9]+)([0-9]{3})/\1,\2/;ta' <<<"${1//_/}"; }
hdur()  { local s=${1%.*}; [[ "$s" =~ ^[0-9]+$ ]] || { echo "-"; return 0; }
          if   (( s < 60 ));    then echo "${s}s"
          elif (( s < 3600 ));  then echo "$((s / 60))m$((s % 60))s"
          elif (( s < 86400 )); then echo "$((s / 3600))h$(((s % 3600) / 60))m"
          else echo "$((s / 86400))d$(((s % 86400) / 3600))h"; fi; }
fmt_prox() { # motes -> "1,234.5" (PROX)
  local m="${1//_/}"; [[ "$m" =~ ^[0-9]+$ ]] || { echo "?"; return 0; }
  awk -v m="$m" -v u="${NET_UNIT:-1000000}" 'BEGIN{printf "%.6f", m/u}' \
    | sed -E ':a;s/^([0-9]+)([0-9]{3})/\1,\2/;ta' | sed -E '/\./{s/0+$//;s/\.$//}'
}
fmt_rate() { awk -v r="${1:-0}" 'BEGIN{ if (r>=1e6) printf "%.2f MH/s", r/1e6; else if (r>=1e3) printf "%.2f KH/s", r/1e3; else printf "%d H/s", r }'; }
spark() { awk 'BEGIN{split("▁ ▂ ▃ ▄ ▅ ▆ ▇ █",b," ")}
  {v[NR]=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1}
  END{for(i=1;i<=NR;i++){k=(mx>mn)?int((v[i]-mn)/(mx-mn)*7)+1:4; if(k>8)k=8; printf "%s",b[k]}}'; }
bar() { # bar <value> <max> <width> <colour>
  local v=$1 m=$2 w=$3 col=$4 f
  f=$(awk -v v="$v" -v m="$m" -v w="$w" 'BEGIN{x=(m>0)?int(v/m*w+0.5):0; if(x>w)x=w; if(x<0)x=0; print x}')
  printf '%s%s%s%s%s' "$col" "$(rep █ "$f")" "$C_D" "$(rep ░ $((w - f)))" "$C_0"
}
stackbar() { # stackbar <width> <a> <b> <c>   (cyan / green / magenta)
  local w=$1 na nb nc
  read -r na nb nc < <(awk -v a="$2" -v b="$3" -v c="$4" -v w="$w" \
    'BEGIN{t=a+b+c; if(t<=0){print 0,0,0; exit} na=int(a/t*w+.5); nb=int(b/t*w+.5); nc=w-na-nb; if(nc<0)nc=0; print na,nb,nc}') || true
  if (( na + nb + nc == 0 )); then printf '%s%s%s' "$C_D" "$(rep ░ "$w")" "$C_0"
  else printf '%s%s%s%s%s%s%s' "$C_C" "$(rep █ "$na")" "$C_G" "$(rep █ "$nb")" "$C_M" "$(rep █ "$nc")" "$C_0"; fi
}
peer_name() { local ip="$1" i; for i in "${!PUBLIC_APIS[@]}"; do
    if [[ "${PUBLIC_APIS[$i]}" == *"//$ip:"* ]]; then echo "${PUBLIC_NAMES[$i]:-public}"; return 0; fi; done; echo ""; }
public_ip() { local u ip
  for u in https://api.ipify.org https://ifconfig.me/ip https://icanhazip.com; do
    ip=$(curl -4fs -m 4 "$u" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then echo "$ip"; return 0; fi
  done; return 0; }
api_host() { awk '/^api:/{f=1;next} f&&/^[^ #]/{f=0} f&&/^  host:/{print $2;exit}' "$NODE_DIR/proxima.yaml" 2>/dev/null || true; }

dash_banner() {
  local title="  PROXIMA DASHBOARD" stamp; stamp="$(date -u '+%F %T') UTC  "
  local pad=$((DASH_W - 4 - ${#title} - ${#stamp} - 2)); if (( pad < 1 )); then pad=1; fi
  printf '\n  %s╭%s╮%s\n' "$C_C" "$(rule $((DASH_W - 4)))" "$C_0"
  printf '  %s│%s %s◆%s%s%s%*s%s%s%s│%s\n' "$C_C" "$C_0" "$C_M$C_B" "$C_0" "$C_B$C_W" "$title" "$pad" "" "$C_D" "$stamp" "$C_C" "$C_0"
  printf '  %s╰%s╯%s\n' "$C_C" "$(rule $((DASH_W - 4)))" "$C_0"
}

# ---- network ---------------------------------------------------------------
dash_network() {
  sect "NETWORK"
  if load_net_info; then
    kv "ledger" "${C_B}${C_W}${NET_DESC:-?}${C_0}"
    kv "genesis" "$(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS") UTC   ${C_D}hash ${NET_HASH:0:12}…${C_0}"
    kv "public nodes run" "${C_C}${NET_NODE_VER:-?}${C_0}   ${C_D}miner version ${NET_MINER_VER:-?}${C_0}"
    local ps; ps=$(sync_info "${PUBLIC_APIS[0]}")
    if [[ -n "$ps" ]]; then
      PUB_LRB=$(jq -r '.lrb_slot // empty' <<<"$ps" 2>/dev/null || true)
      kv "current slot" "${C_W}$(jq -r '.current_slot' <<<"$ps" 2>/dev/null || echo '?')${C_0}   ${C_D}network LRB slot ${PUB_LRB:-?}${C_0}"
    fi
    if [[ "$HAVE_NODE" == 1 && "$(db_ledger_state)" == other ]]; then
      printf '  %s%s ✘ this node'"'"'s DB is from an OLD ledger — run: %s reset %s\n' "$C_B" "$C_R" "$0" "$C_0"
    fi
  else
    printf '  %s! public nodes not reachable%s\n' "$C_Y" "$C_0"
  fi
}

# ---- this node: address, sync, peers ---------------------------------------
svc_state() { # svc_state node|miner -> active|inactive|running|stopped|none
  local n="$1" unit="$NODE_SVC"; [[ "$n" == miner ]] && unit="$MINER_SVC"
  if have_systemd && [[ -f "/etc/systemd/system/$unit.service" ]]; then systemctl is-active "$unit" 2>/dev/null || true; return 0; fi
  local pf="$PROXIMA_HOME/$n.pid"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo running; return 0; fi
  if [[ -n "$(find_proc "$n")" ]]; then echo running; return 0; fi     # started by hand
  # installed but not running (a plain 'stop' removes the pid file)
  if [[ -f "$pf" ]]; then echo stopped; return 0; fi
  if [[ "$n" == node && -f "$NODE_DIR/proxima.yaml" ]]; then echo stopped; return 0; fi
  if [[ "$n" == miner && -f "$WALLET_DIR/miner.log" ]]; then echo stopped; return 0; fi
  echo none
}
find_proc() { # pid of a running node / miner, whatever started it
  case "$1" in
    node)  pgrep -n -f "^$BIN_DIR/proxima( |\$)" 2>/dev/null || true;;
    miner) pgrep -n -f "^[^ ]*/proxi node mine" 2>/dev/null || true;;
  esac
}
svc_pid() {
  local n="$1" unit="$NODE_SVC" p=""; [[ "$n" == miner ]] && unit="$MINER_SVC"
  if have_systemd && [[ -f "/etc/systemd/system/$unit.service" ]]; then p=$(systemctl show -p MainPID --value "$unit" 2>/dev/null || true)
  elif [[ -f "$PROXIMA_HOME/$n.pid" ]]; then p=$(cat "$PROXIMA_HOME/$n.pid" 2>/dev/null || true); fi
  if [[ ! "$p" =~ ^[0-9]+$ ]] || ! kill -0 "$p" 2>/dev/null; then p=$(find_proc "$n"); fi
  if [[ "$p" =~ ^[0-9]+$ && "$p" != 0 ]]; then echo "$p"; fi
  return 0
}
svc_badge() { case "$1" in
    active|running) printf '%s●%s %s%s%s' "$C_G" "$C_0" "$C_G" "$1" "$C_0";;
    activating)     printf '%s◐%s %s%s%s' "$C_Y" "$C_0" "$C_Y" "$1" "$C_0";;
    none)           printf '%s○ not installed%s' "$C_D" "$C_0";;
    *)              printf '%s○%s %s%s%s' "$C_R" "$C_0" "$C_R" "$1" "$C_0";; esac; }
proc_stats() { # pid -> "up 2h13m · 312 MB RSS · 14.2% CPU"
  local pid="$1" et rss cpu; [[ -n "$pid" ]] || return 0
  read -r et rss cpu < <(ps -o etimes=,rss=,pcpu= -p "$pid" 2>/dev/null) || return 0
  printf 'up %s · %s MB RAM · %s%% CPU' "$(hdur "$et")" "$((rss / 1024))" "$cpu"
}

dash_node() {
  sect "THIS NODE"
  if [[ "$HAVE_NODE" != 1 ]]; then
    kv "mode" "${C_D}no local node — wallet and miner use ${MINER_API:-a public node}${C_0}"; return 0
  fi
  local api; api=$(local_api)
  local ni si st; ni=$(curl -fs -m 5 "$api/api/v1/node_info" 2>/dev/null || true); si=$(sync_info "$api")
  st=$(svc_state node)
  kv "service" "$(svc_badge "$st")   ${C_D}$(proc_stats "$(svc_pid node)")${C_0}"

  if [[ -z "$ni" ]]; then
    kv "api" "${C_R}not reachable at $api${C_0} ${C_D}(starting up or restoring the snapshot)${C_0}"
    local last; last=$(tail -n 2 "$NODE_DIR/proxima.log" 2>/dev/null | cut -c1-110 || true)
    [[ -n "$last" ]] && while IFS= read -r l; do note "log: $l"; done <<<"$last"
    return 0
  fi
  local id ver commit stress pipe syncing nst ndy
  IFS=$'\t' read -r id ver commit stress pipe syncing nst ndy < <(jq -r \
    '[.id,.version,(.commit_hash[0:8]),(.memory_stress_level//0),(.pipeline_size//0),(.is_syncing//false),(.num_static_peers//0),(.num_dynamic_alive//0)]|@tsv' <<<"$ni" 2>/dev/null) || true

  local synced cur lrb; synced=$(jq -r '.synced' <<<"$si" 2>/dev/null || echo false)
  cur=$(jq -r '.current_slot // 0' <<<"$si" 2>/dev/null || echo 0); lrb=$(jq -r '.lrb_slot // 0' <<<"$si" 2>/dev/null || echo 0)
  local behind=$((cur - lrb)) lag=""
  if [[ "${PUB_LRB:-}" =~ ^[0-9]+$ ]]; then lag="   ${C_D}network LRB ${PUB_LRB} → $((PUB_LRB - lrb)) slots behind${C_0}"; fi
  if [[ "$synced" == true ]]; then
    kv "sync" "$(dot_on) ${C_G}${C_B}SYNCED${C_0}   ${C_D}slot${C_0} ${C_W}$cur${C_0} ${C_D}· LRB${C_0} ${C_W}$lrb${C_0} ${C_D}(${behind} behind)${C_0}$lag"
  else
    kv "sync" "$(dot_mid) ${C_Y}${C_B}SYNCING${C_0}   ${C_D}slot${C_0} ${C_W}$cur${C_0} ${C_D}· LRB${C_0} ${C_W}$lrb${C_0} ${C_Y}(${behind} behind)${C_0}$lag"
  fi
  kv "version" "${C_C}${ver:-?}${C_0} ${C_D}(${commit:-?})${C_0}"

  # --- addresses
  local pip pport aport ahost lips
  pip=$(public_ip); pport=$(peering_port); aport=$(api_port); ahost=$(api_host)
  lips=$(hostname -I 2>/dev/null | xargs || true)
  kv "host (peer) ID" "${C_W}$id${C_0}"
  if [[ -n "$pip" ]]; then
    kv "public IP" "${C_B}${C_G}$pip${C_0}   ${C_D}local: ${lips:-?}${C_0}"
    kv "peering (UDP)" "${C_B}${C_C}$pip:$pport${C_0}   ${C_D}QUIC — must be open inbound${C_0}"
    kv "multiaddr" "${C_D}/ip4/${C_0}$pip${C_D}/udp/$pport/quic-v1/p2p/${C_0}${C_D}$id${C_0}"
    if [[ " $lips " != *" $pip "* ]]; then note "                    public IP is not on a local interface (NAT?): forward UDP $pport"; fi
    if [[ -n "$ahost" && "$ahost" == 127.* ]]; then
      kv "API (TCP)" "${C_B}${C_C}127.0.0.1:$aport${C_0}   ${C_D}local only${C_0}"
    else
      kv "API (TCP)" "${C_B}${C_C}$pip:$aport${C_0}   ${C_D}http://$pip:$aport/dashboard${C_0}"
    fi
  else
    kv "peering (UDP)" "${C_B}${C_C}<this-host>:$pport${C_0}   ${C_D}public IP lookup failed${C_0}"
    kv "API (TCP)" "${C_B}${C_C}<this-host>:$aport${C_0}"
  fi
  local sc=$C_G; if (( ${stress:-0} >= 3 )); then sc=$C_R; elif (( ${stress:-0} >= 2 )); then sc=$C_Y; fi
  kv "load" "${C_D}memory stress${C_0} ${sc}${stress:-0}${C_0}  ${C_D}· pipeline${C_0} ${C_W}$(thou "${pipe:-0}")${C_0}"
  # --- disk
  local kb; kb=$(du -sck "$NODE_DIR"/proximadb* 2>/dev/null | tail -1 | awk '{print $1}' || true)
  local free; free=$(df -Pk "$NODE_DIR" 2>/dev/null | awk 'NR==2{printf "%.1f GB free", $4/1048576}' || true)
  if [[ "$kb" =~ ^[0-9]+$ ]]; then kv "database" "${C_W}$(awk -v k="$kb" 'BEGIN{ if (k>=1048576) printf "%.2f GB", k/1048576; else printf "%.0f MB", k/1024 }')${C_0}   ${C_D}$free${C_0}"; fi

  # --- peers
  local pi; pi=$(curl -fs -m 6 "$api/api/v1/peers_info" 2>/dev/null || true)
  dash_peers "$pi" 14 "$nst" "$ndy"
}

dash_peers() { # <peers_info json> <max rows> [static count] [dyn alive]
  local pi="$1" max="${2:-14}" total=0 alive=0 sumrtt=0 nrtt=0
  sect "CONNECTED PEERS"
  if [[ -z "$pi" ]] || ! jq -e '.peers' >/dev/null 2>&1 <<<"$pi"; then
    if [[ -n "$pi" ]]; then printf '  %s○ no peers connected yet%s\n' "$C_Y" "$C_0"
    else printf '  %s! peers_info not available%s\n' "$C_Y" "$C_0"; fi
    return 0
  fi
  local rows; rows=$(jq -r '(.peers // []) | sort_by((if .is_static then 0 else 1 end), (if .is_alive then 0 else 1 end), (.rtt_ms // 99999)) | .[]
    | def hp: split("/") | if length>4 then (if .[1]=="ip6" then "["+.[2]+"]" else .[2] end)+":"+.[4] else "-" end;
      ((.multiAddresses // []) as $m | (([$m[]|select(startswith("/ip4/"))]+$m)[0] // "") | hp) as $a
    | [.id, $a, .is_static, .is_alive, (.rtt_ms // -1 | floor), (.num_incoming_tx // 0), (((now - (.when_added // 0)/1e9)) | floor)] | @tsv' <<<"$pi" 2>/dev/null || true)
  total=$(wc -l <<<"$rows" | tr -d ' '); [[ -n "$rows" ]] || total=0
  local id addr stat al rtt tx age n=0 nstat=0 nstat_alive=0 best=999999
  while IFS=$'\t' read -r id addr stat al rtt tx age; do
    [[ -n "$id" ]] || continue
    [[ "$al" == true ]] && alive=$((alive + 1))
    if [[ "$stat" == true ]]; then nstat=$((nstat + 1)); [[ "$al" == true ]] && nstat_alive=$((nstat_alive + 1)); fi
    if [[ "$al" == true && "$rtt" -ge 0 ]]; then sumrtt=$((sumrtt + rtt)); nrtt=$((nrtt + 1)); (( rtt < best )) && best=$rtt; fi
  done <<<"$rows"
  local avg="-"; (( nrtt > 0 )) && avg=$((sumrtt / nrtt))
  local hc=$C_G; if (( total == 0 || alive == 0 )); then hc=$C_R; elif (( alive < 3 )); then hc=$C_Y; fi
  printf '  %s%s%s %s%s%s %salive of%s %s%s%s   %s%s static · %s autopeered%s' \
    "$hc" "$(bar "$alive" "$(( total > 0 ? total : 1 ))" 16 "$hc")" "$C_0" "$C_B$hc" "$alive" "$C_0" "$C_D" "$C_0" "$C_W" "$total" "$C_0" "$C_D" "$nstat_alive/$nstat" "$((alive - nstat_alive))" "$C_0"
  if (( nrtt > 0 )); then printf '   %savg RTT %s ms · best %s ms%s' "$C_D" "$avg" "$best" "$C_0"; fi
  printf '\n\n'
  printf '  %s  %-6s %-6s %-12s %-21s %6s %7s %6s%s\n' "$C_D" "TYPE" "NAME" "PEER ID" "ADDRESS (UDP)" "RTT" "TX-IN" "AGE" "$C_0"
  while IFS=$'\t' read -r id addr stat al rtt tx age; do
    [[ -n "$id" ]] || continue
    if (( n >= max )); then note "  … and $((total - max)) more — run: $0 peers"; break; fi
    n=$((n + 1))
    local d tcol=$C_D tname="auto" rc=$C_G nm ip rt="-" sid
    if [[ "$al" == true ]]; then d=$(dot_on); else d=$(dot_off); fi
    if [[ "$stat" == true ]]; then tcol=$C_BL; tname="static"; fi
    ip=${addr%:*}; ip=${ip#[}; ip=${ip%]}; nm=$(peer_name "$ip")
    if (( rtt >= 0 )); then rt="${rtt} ms"; fi
    if (( rtt >= 250 )); then rc=$C_R; elif (( rtt >= 100 )); then rc=$C_Y; fi
    [[ "$al" == true ]] || rc=$C_D
    sid="${id:8:5}…${id: -6}"
    printf '  %s %s %s %s%s%s %s %s %s %s\n' "$d" "$(pc "$tcol" 6 "$tname")" "$(pc "$C_M$C_B" 6 "$nm")" \
      "$C_W" "$sid" "$C_0" "$(pc "$C_C" 21 "$addr")" "$(pcr "$rc" 6 "$rt")" "$(pcr "$C_D" 7 "$(thou "$tx")")" "$(pcr "$C_D" 6 "$(hdur "$age")")"
  done <<<"$rows"
  return 0
}

# ---- wallet: balance + delegations -----------------------------------------
dash_wallet() {
  sect "WALLET"
  if [[ ! -f "$WALLET_DIR/proxi.yaml" ]]; then kv "wallet" "${C_D}not configured${C_0}"; return 0; fi
  local tb td; tb=$(mktemp); td=$(mktemp)
  ( cd "$WALLET_DIR" && timeout 40 "$BIN_DIR/proxi" node balance </dev/null >"$tb" 2>&1 || true ) &
  ( cd "$WALLET_DIR" && timeout 40 "$BIN_DIR/proxi" node delegate status </dev/null >"$td" 2>&1 || true ) &
  wait
  local B; B=$(tr -d '\r' <"$tb"); local D; D=$(tr -d '\r' <"$td"); rm -f "$tb" "$td"

  local acct; acct=$(sed -nE 's/^ *wallet account[^:]*: *([^ ]+).*/\1/p' <<<"$B" | head -1 || true)
  [[ -n "$acct" ]] && kv "account" "${C_W}$acct${C_0}"
  if ! grep -q 'SUMMARY controlled' <<<"$B"; then
    local msg; msg=$(grep -vE '^(Command line|config profile)' <<<"$B" | tail -n 2 | cut -c1-100 || true)
    kv "balance" "${C_R}unavailable${C_0} ${C_D}(node not synced yet, or the key is encrypted)${C_0}"
    [[ -n "$msg" ]] && while IFS= read -r l; do note "  $l"; done <<<"$msg"
    return 0
  fi
  local nc_n nc_v dl_n dl_v ch_n ch_v tot_n tot_v
  read -r nc_n nc_v < <(sed -nE 's/^ *on +([0-9]+) non-chain outputs: +([0-9_]+).*/\1 \2/p' <<<"$B" | head -1) || true
  read -r dl_n dl_v < <(sed -nE 's/^ *on +([0-9]+) delegation outputs: +([0-9_]+).*/\1 \2/p' <<<"$B" | head -1) || true
  read -r ch_n ch_v < <(sed -nE 's/^ *on +([0-9]+) non-delegation chain outputs: +([0-9_]+).*/\1 \2/p' <<<"$B" | head -1) || true
  read -r tot_n tot_v < <(sed -nE 's/^ *TOTAL controlled on ([0-9]+) outputs: +([0-9_]+).*/\1 \2/p' <<<"$B" | head -1) || true
  nc_v=${nc_v//_/}; dl_v=${dl_v//_/}; ch_v=${ch_v//_/}; tot_v=${tot_v//_/}
  local tick="${NET_TICKER:-PROX}"
  kv "TOTAL" "${C_B}${C_G}$(fmt_prox "${tot_v:-0}") $tick${C_0}   ${C_D}on ${tot_n:-0} outputs${C_0}"
  printf '  %s%-17s%s %s\n' "$C_D" "" "$C_0" "$(stackbar 44 "${nc_v:-0}" "${dl_v:-0}" "${ch_v:-0}")"
  kv "${C_C}■${C_0}${C_D} liquid${C_0}" "${C_W}$(fmt_prox "${nc_v:-0}") $tick${C_0}   ${C_D}${nc_n:-0} plain outputs${C_0}"
  kv "${C_G}■${C_0}${C_D} delegated${C_0}" "${C_W}$(fmt_prox "${dl_v:-0}") $tick${C_0}   ${C_D}${dl_n:-0} delegations${C_0}"
  kv "${C_M}■${C_0}${C_D} other chains${C_0}" "${C_W}$(fmt_prox "${ch_v:-0}") $tick${C_0}   ${C_D}${ch_n:-0} chain outputs${C_0}"
  local lrbs; lrbs=$(sed -nE 's/^ *Latest reliable branch \(LRB\) txid: *([^ ]+).*/\1/p' <<<"$B" | head -1 || true)
  [[ -n "$lrbs" ]] && kv "as of LRB" "${C_D}${lrbs:0:44}…${C_0}"

  # ---- delegations
  local drows; drows=$(awk '/^ *DELEGATIONS \(/{f=1;next} f&&/^[A-Z]/{f=0} f{id=$1; sub(/^\$\//,"",id); if (length(id)>=20 && id ~ /^[0-9a-fA-F]+$/) print}' <<<"$B" || true)
  sect "DELEGATIONS"
  if [[ -z "$drows" ]]; then
    printf '  %s○ none yet%s %s— mining rewards are paid into delegations once the miner has won transits%s\n' "$C_Y" "$C_0" "$C_D" "$C_0"
    return 0
  fi
  local tmap; tmap=$(grep -E '^ *(\$/)?[0-9a-fA-F]{20,} .* -> ' <<<"$D" | awk '{print $1" "$NF}' || true)
  printf '  %s  %-3s %-19s %17s  %-36s %s%s\n' "$C_D" "#" "CHAIN ID" "BALANCE" "STATUS" "TARGET" "$C_0"
  local i=0 id bal rest status tgt sc
  while read -r id bal rest; do
    [[ -n "$id" ]] || continue
    i=$((i + 1))
    local full="$id"; id=${id#\$/}
    status=${rest%% maxFrozen*}; status=${status%,}
    status=$(sed -E 's/\(([0-9]+) hours?, ([0-9]+) min from now\)/(\1h\2m)/; s/\(([0-9]+) min from now\)/(\1m)/; s/until slot/→ slot/; s/for ([0-9]+) min more/\1m left/' <<<"$status")
    tgt=$(awk -v i="$full" '$1==i{print $2;exit}' <<<"$tmap" || true); tgt=${tgt#\$/}
    case "$status" in
      frozen*)           sc=$C_Y;; "safe revocation"*) sc=$C_M;; "can be unlocked"*) sc=$C_G;; "on hold"*) sc=$C_R;; *) sc=$C_W;; esac
    printf '  %s  %s %s%s…%s%s %s %s  %s %s%s%s\n' "$C_D" "$(printf '%-3s' "$i")" "$C_W" "${id:0:12}" "${id: -6}" "$C_0" \
      "$(pcr "$C_G$C_B" 17 "$(fmt_prox "$bal")")" "$C_D$tick$C_0" "$(pc "$sc" 36 "${status:0:36}")" "$C_C" "${tgt:+${tgt:0:10}…${tgt: -6}}" "$C_0"
  done <<<"$drows"
  return 0
}

# ---- miner: hashrate --------------------------------------------------------
miner_log() {
  local o=""
  if have_systemd && [[ -f "/etc/systemd/system/$MINER_SVC.service" ]]; then
    o=$(journalctl -u "$MINER_SVC" --no-pager -n 1500 -o cat 2>/dev/null || true)
    [[ -n "$o" ]] || o=$(sudo -n journalctl -u "$MINER_SVC" --no-pager -n 1500 -o cat 2>/dev/null || true)
  else
    o=$(tail -c 80000 "$WALLET_DIR/miner.log" 2>/dev/null || true)
  fi
  printf '%s' "$o" | tr '\r' '\n'
}

dash_miner() {
  sect "MINER"
  local st; st=$(svc_state miner)
  local pid cmd; pid=$(svc_pid miner); cmd=""
  if [[ -n "$pid" ]]; then cmd=$(ps -o args= -p "$pid" 2>/dev/null || true); fi
  kv "service" "$(svc_badge "$st")   ${C_D}$(proc_stats "$pid")${C_0}"
  if [[ "$st" == none ]]; then return 0; fi

  local workers seeker cap
  workers=$(grep -oE -- '--workers +[0-9]+' <<<"$cmd" | awk '{print $2}' || true); [[ -n "$workers" ]] || workers="all $(nproc 2>/dev/null || echo '?') cores"
  seeker="no"; if grep -q -- '--seeker' <<<"$cmd"; then seeker="${C_G}yes (Rust nonce_seeker)${C_0}"; fi
  cap=$(grep -oE -- '--max-hashrate-khs +[0-9.]+' <<<"$cmd" | awk '{print $2" KH/s cap"}' || true)
  kv "workers" "${C_W}$workers${C_0}   ${C_D}seeker:${C_0} $seeker${cap:+   ${C_D}$cap${C_0}}"

  local log; log=$(miner_log)
  if [[ -z "$log" ]]; then
    kv "hashrate" "${C_Y}no log readable${C_0} ${C_D}(run as root to read the miner journal)${C_0}"; return 0
  fi
  local rate="" samples="" tot="" k="" solved=0 improved=0 err="" tslot="" lastn=""
  local key val
  while IFS=$'\t' read -r key val; do
    case "$key" in rate) rate=$val;; samples) samples=$val;; tot) tot=$val;; k) k=$val;; solved) solved=$val;;
                   improved) improved=$val;; err) err=$val;; tslot) tslot=$val;; lastn) lastn=$val;; esac
  done < <(awk '
    { if (match($0,/mining\.\.\. [0-9_]+ attempts +[0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,f," "); r=f[4]; gsub(/_/,"",r); sm[++ns]=r }
      if (match($0,/attempts at [0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,f," "); r=f[3]; gsub(/_/,"",r); if (r+0>0) sm[++ns]=r }
      if (index($0,"totals:")) tot=substr($0,index($0,"totals:"))
      if (match($0,/difficulty K=[0-9]+/)) k=substr($0,RSTART+13,RLENGTH-13)
      if (match($0,/target slot [0-9]+/)) ts=substr($0,RSTART+12,RLENGTH-12)
      if (match($0,/mining transit #[0-9]+/)) ln=substr($0,RSTART+16,RLENGTH-16)
      if (index($0,"SOLVED transit")) sv++
      if (index($0,"IMPROVED transit")) im++
      if (index($0,"submit failed")||index($0,"not answering")||index($0,"Error:")) er=$0 }
    END { if (ns>0) { print "rate\t" sm[ns]; s=""; st=(ns>40)?ns-39:1; for(i=st;i<=ns;i++) s=s sm[i] " "; print "samples\t" s }
          print "tot\t" tot; print "k\t" k; print "solved\t" sv+0; print "improved\t" im+0; print "err\t" er; print "tslot\t" ts; print "lastn\t" ln }' <<<"$log") || true

  if [[ "$rate" =~ ^[0-9]+$ ]]; then
    local avg peak; read -r avg peak < <(awk '{n=0;s=0;m=0; for(i=1;i<=NF;i++){ if(NF>5 && i<=2) continue; s+=$i;n++; if($i>m)m=$i } printf "%d %d", (n?s/n:0), m}' <<<"$samples") || true
    kv "hashrate" "${C_B}${C_G}$(fmt_rate "$rate")${C_0}   ${C_D}avg${C_0} ${C_W}$(fmt_rate "${avg:-0}")${C_0}  ${C_D}peak${C_0} ${C_W}$(fmt_rate "${peak:-0}")${C_0}"
    local sp; sp=$(tr ' ' '\n' <<<"$samples" | awk 'NF' | spark)
    [[ -n "$sp" ]] && kv "recent" "${C_C}$sp${C_0}"
  else
    kv "hashrate" "${C_Y}no samples yet${C_0} ${C_D}(waiting for the node to sync / next mining round)${C_0}"
  fi

  # --- totals reported by the miner itself
  local re='confirmed ([0-9]+) \(\+([0-9]+) in flight, ([0-9]+) orphaned, ([0-9]+) tracked\) \| minted ([0-9_]+) \| K=([0-9]+) \| attempts ([0-9_]+) \| avg ([0-9_]+) H/s \| uptime ([^ ]+)'
  if [[ "$tot" =~ $re ]]; then
    kv "transits" "${C_G}${C_B}${BASH_REMATCH[1]}${C_0} ${C_D}confirmed${C_0}  ${C_Y}${BASH_REMATCH[2]}${C_0} ${C_D}in flight${C_0}  ${C_R}${BASH_REMATCH[3]}${C_0} ${C_D}orphaned${C_0}"
    kv "minted" "${C_B}${C_G}$(fmt_prox "${BASH_REMATCH[5]}") ${NET_TICKER:-PROX}${C_0}   ${C_D}attempts $(thou "${BASH_REMATCH[7]}")  ·  avg $(thou "${BASH_REMATCH[8]}") H/s  ·  session ${BASH_REMATCH[9]}${C_0}"
    [[ -n "$k" ]] || k=${BASH_REMATCH[6]}
  else
    kv "transits" "${C_D}none confirmed yet in the visible log${C_0}"
  fi
  if [[ "$k" =~ ^[0-9]+$ ]]; then
    kv "difficulty" "${C_W}K=$k${C_0} ${C_D}(a nonce needs $k trailing zero bits)${C_0}${lastn:+   ${C_D}working on transit #$lastn${tslot:+, slot $tslot}${C_0}}"
    if [[ "$rate" =~ ^[0-9]+$ && "$rate" -gt 0 ]]; then
      local est; est=$(awk -v r="$rate" -v k="$k" -v s="${NET_SLOT_SEC:-10.24}" -v rw="${NET_REWARD:-95000000}" -v u="${NET_UNIT:-1000000}" \
        'BEGIN{ n=2^k; p=1-exp(-r*s/n); if (p<=0) p=1e-12; t=s/p; perday=p*86400/s;   # at most one step per slot
                printf "%d %.4f %.2f %.1f", t, p*100, perday, perday*rw/u }') || true
      local t_s p_s pd rwd; read -r t_s p_s pd rwd <<<"$est"
      kv "best case" "${C_W}1 solution per ~$(hdur "$t_s")${C_0} ${C_D}· ${p_s}% per slot · at most ≈ ${rwd} ${NET_TICKER:-PROX}/day${C_0}"
      note "                    (upper bound: assumes no rival miners and no contested slots; real income is lower)"
    fi
  fi
  if (( solved + improved > 0 )); then kv "recent finds" "${C_G}${solved}${C_0} ${C_D}solved ·${C_0} ${C_G}${improved}${C_0} ${C_D}improved (in visible log)${C_0}"; fi
  if [[ -n "$err" ]]; then kv "last problem" "${C_Y}${err:0:90}${C_0}"; fi
}

dash_software() {
  sect "SOFTWARE"
  local iv pv; iv=$(installed_version); pv=$("$BIN_DIR/proxi" version 2>/dev/null | awk '/Version:/{print $2}' || true)
  kv "installed" "proxima ${C_C}${iv:-?}${C_0}  ${C_D}·${C_0}  proxi ${C_C}${pv:-?}${C_0}$([[ -x $BIN_DIR/nonce_seeker ]] && echo "  ${C_D}·${C_0}  nonce_seeker ${C_G}yes${C_0}")"
  if [[ -d "$SRC_DIR/.git" ]]; then
    git -C "$SRC_DIR" fetch --quiet --tags origin 2>/dev/null || true
    local ur="origin/$BRANCH"
    git -C "$SRC_DIR" rev-parse --verify --quiet "refs/remotes/$ur" >/dev/null 2>&1 || ur="$BRANCH"
    local rv; rv=$(git -C "$SRC_DIR" show "$ur:global/version.go" 2>/dev/null | grep -oE 'Version\s+=\s+"[^"]+"' | cut -d'"' -f2 || true)
    local rh; rh=$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || true); local oh; oh=$(git -C "$SRC_DIR" rev-parse "$ur^{commit}" 2>/dev/null || true)
    if [[ -n "$rv" && ( "$rv" != "$iv" || "$rh" != "$oh" ) ]]; then
      kv "upstream" "${C_Y}$BRANCH has ${rv} (${oh:0:8}) — update available: ${C_B}$0 update${C_0}"
    else kv "upstream" "${C_G}up to date${C_0} ${C_D}($BRANCH ${rv:-?})${C_0}"; fi
  fi
}

dash_footer() {
  local me="${0##*/}" d="${C_D}·${C_0}"
  printf '\n  %s' "$C_D"; rule $((DASH_W - 4)); printf '%s\n' "$C_0"
  printf '  %s%s logs miner|node%s %s %s%s peers%s %s %s%s watch [secs]%s %s %s%s update%s %s %s%s reset%s\n' \
    "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0"
}

cmd_status() {
  HAVE_NODE=0; PUB_LRB=""
  if [[ "$NO_NODE" != 1 && -f "$NODE_DIR/proxima.yaml" ]]; then HAVE_NODE=1; fi
  [[ -n "$MINER_API" ]] || MINER_API=$(awk '/node_url:/{print $2;exit}' "$WALLET_DIR/proxi.yaml" 2>/dev/null || true)
  dash_banner
  dash_network
  dash_node
  dash_wallet
  dash_miner
  dash_software
  dash_footer
}

# all peers, full IDs and every multiaddress
cmd_peers() {
  [[ -f "$NODE_DIR/proxima.yaml" ]] || die "no local node installed here"
  local pi; pi=$(curl -fs -m 8 "$(local_api)/api/v1/peers_info" 2>/dev/null || true)
  [[ -n "$pi" ]] || die "node API not reachable at $(local_api)"
  dash_banner
  sect "PEERS (full)"
  printf '  %s%s%s\n' "$C_D" "host ID  $(jq -r .host_id <<<"$pi")" "$C_0"
  local rows; rows=$(jq -c '(.peers // []) | sort_by((if .is_static then 0 else 1 end), (if .is_alive then 0 else 1 end), (.rtt_ms // 99999)) | .[]' <<<"$pi")
  local n=0 line
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    n=$((n + 1))
    local id al stat rtt tx pull age d tc="auto" ip nm
    id=$(jq -r .id <<<"$line"); al=$(jq -r .is_alive <<<"$line"); stat=$(jq -r .is_static <<<"$line")
    rtt=$(jq -r '(.rtt_ms // -1) | floor' <<<"$line"); tx=$(jq -r '.num_incoming_tx // 0' <<<"$line"); pull=$(jq -r '.num_incoming_pull // 0' <<<"$line")
    age=$(jq -r '((now - (.when_added // 0)/1e9) | floor)' <<<"$line")
    if [[ "$al" == true ]]; then d=$(dot_on); else d=$(dot_off); fi
    [[ "$stat" == true ]] && tc="static"
    ip=$(jq -r '((.multiAddresses // [])[]|select(startswith("/ip4/")))' <<<"$line" | head -1 | awk -F/ '{print $3}' || true)
    nm=$(peer_name "${ip:-x}")
    printf '\n  %s %s%s%2d%s  %s%s%s %s%s%s\n' "$d" "$C_D" "" "$n" "$C_0" "$C_W" "$id" "$C_0" "$C_M$C_B" "$nm" "$C_0"
    printf '      %s%-6s%s  rtt %s  ·  tx-in %s  ·  pull-in %s  ·  connected %s%s\n' "$C_D" "$tc" "$C_0" "$( ((rtt >= 0)) && echo "${rtt} ms" || echo -)" "$(thou "$tx")" "$(thou "$pull")" "$(hdur "$age")" "$C_0"
    jq -r '(.multiAddresses // [])[]' <<<"$line" | while IFS= read -r a; do printf '      %s%s%s\n' "$C_C" "$a" "$C_0"; done
  done <<<"$rows"
  (( n > 0 )) || printf '  %sno peers connected%s\n' "$C_Y" "$C_0"
  echo
}

# live dashboard, refreshed in place
cmd_watch() {
  local n="${1:-10}"; [[ "$n" =~ ^[0-9]+$ && "$n" -ge 3 ]] || n=10
  [[ -t 1 ]] || die "watch needs a terminal"
  tput civis 2>/dev/null || true
  trap 'tput cnorm 2>/dev/null || true; printf "\n"; exit 0' INT TERM
  local out
  while true; do
    out=$(cmd_status 2>&1 | sed 's/$/\x1b[K/' || true)
    printf '\e[H%s\n\n  %srefreshing every %ss — Ctrl-C to quit%s\e[K\n\e[J' "$out" "$C_D" "$n" "$C_0"
    sleep "$n"
  done
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
  peers)     cmd_peers;;
  watch|dashboard) cmd_watch "${ARGS[0]:-10}";;
  build)     cmd_build;;
  start|stop|restart) svc "$CMD" "${ARGS[0]:-all}";;
  status)    cmd_status;;
  logs)      cmd_logs "${ARGS[0]:-miner}";;
  remote)    cmd_remote;;
  wait-sync) cmd_wait_sync "${ARGS[0]:-}";;
  uninstall) cmd_uninstall;;
  help|*)    usage;;
esac
