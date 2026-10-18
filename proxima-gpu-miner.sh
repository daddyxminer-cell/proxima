#!/usr/bin/env bash
# =============================================================================
#  proxima-gpu-miner.sh — standalone CPU + NVIDIA GPU Proxima miner for a VPS / rig
#
#  Mines on THIS machine and talks to a node that already runs (and is synced)
#  somewhere else.  No node, no database: it builds `proxi` from
#  https://github.com/lunfardo314/proxima (master) and runs `proxi node mine --seeker`
#  with a CUDA nonce seeker (`gpu_seeker`, sources shipped inside this script).
#  Target network: Proxima "take 1" (v1.0.x-take1, miner version 4).
#  Built for proxi v1.0.1-take1 (the miner extends the canonical branch, not the tallest).
#  Note: the key is held in the seeker's memory while it mines - use a dedicated wallet on
#  machines you control (see proxi/node_cmd/mine/nonce_seeker/README.md upstream).
#  Same configuration, prompts, flags and commands as proxima-miner.sh (CPU), plus GPU options.
#
#  GPUs: written for NVIDIA RTX 4000 (Ada, sm_89) and RTX 5000 (Blackwell, sm_120); other
#  cards with compute capability >= 7.5 compile too but are untested.  Needs the NVIDIA
#  driver (libcuda; >= 570 for RTX 50xx).  The CUDA compiler is found on the machine
#  or downloaded privately (no root, nothing installed system-wide).  No NVIDIA GPU
#  or driver?  The script then installs a plain CPU miner.
#
#  `install` asks for what it needs:
#    1. the API URL of your synced node          (default: http://161.97.92.49:8000)
#    2. the wallet key that receives the rewards (import yours, or generate one)
#    3. GPU on/off, which GPUs, intensity        (only when an NVIDIA GPU is found)
#    4. how many CPU threads to mine with besides the GPU (0 = GPU only)
#
#  WHERE DO THE REWARDS GO?  Mining is non-outsourceable: the mine covenant pins
#  the payout to the key that signs the mining transaction, so a reward can only
#  land in the wallet of the key this machine mines with.
#    * import the SAME proxima.key as your main wallet -> rewards go straight to
#      your main wallet (several machines under one key search disjoint nonces)
#    * or let the script generate a new key -> rewards go to that new wallet;
#      move them later with:  proxi node send_to_wallet <amount> <holder ID>
#
#  Commands:
#    install       ask for node / key / GPU / threads, build, install, start
#    status        colour dashboard: node link + sync, wallet, balance,
#                  delegations, hashrate (total + GPU), GPU load/temp/power, difficulty
#    watch [secs]  the same dashboard, refreshed in place (default 10 s)
#    check         connectivity test: this machine -> your node, plus GPU / driver /
#                  CUDA / kernel self-test (works before install)
#    gpu [on|off|test]  show the GPU setup; switch the GPU on/off; run the kernel self-test.
#                  With --gpu-devices / --gpu-intensity it changes them and restarts.
#    bench [secs]  stop the miner, measure the hashrate of every GPU, restart
#    logs          follow the miner log
#    start | stop | restart
#    set-node URL  point the miner at another node (keeps the key), restart
#    key           change the wallet key (import / paste / generate), restart
#    update        pull master, rebuild proxi + the GPU seeker, restart (key + config kept)
#    uninstall     stop + remove service and binaries (the wallet is kept)
#    help
#
#  Options (flags, or the environment variable in brackets):
#    --node-url URL     [NODE_URL]       your node's API, e.g. http://1.2.3.4:8000
#    --key-file FILE    [KEY_FILE]       import this proxima.key (no prompt)
#    --generate-key     [GENERATE_KEY=1] create a new key (no prompt)
#    --workers N        [MINER_WORKERS]  CPU threads (0 = GPU only; default 0 with a GPU,
#                                        all cores without)
#    --max-khs X        [MINER_MAX_KHS]  hashrate cap in KH/s for the CPU threads (0 = none);
#                                        the GPU is limited with --gpu-intensity instead
#    --gpu | --no-gpu   [GPU=1|0]        force the GPU on / CPU only (default: use it if found)
#    --gpu-devices L    [GPU_DEVICES]    comma list of CUDA device numbers (default: all)
#    --gpu-intensity I  [GPU_INTENSITY]  low | normal | high   (default normal)
#                                          low    short kernels, desktop stays smooth, ~lower rate
#                                          normal ~150 ms kernels, 4 blocks per SM
#                                          high   ~300 ms kernels, 8 blocks per SM
#    --gpu-arch "89 120" [GPU_ARCHS]     CUDA architectures to compile for (default: the
#                                        installed GPUs; 89 120 when none is visible)
#    --no-public        [NO_PUBLIC=1]    talk ONLY to your node (no public-node
#                                        witnesses / extra mining streams)
#    --home DIR         [MINER_HOME]     base directory (~/proxima-miner)
#    --branch REF       [BRANCH]         proxima git branch, tag or commit (master);
#                                        pin a commit when master is ahead of the network
#    --bin-dir DIR      [BIN_DIR]        where proxi + gpu_seeker go (<home>/bin)
#    --no-services      [NO_SERVICES=1]  no systemd: run as a background process
#    --no-color         [NO_COLOR=1]     plain output (automatic when piped)
#    -y | --yes         [ASSUME_YES=1]   take defaults, never prompt
#
#  Already running the CPU miner (proxima-miner.sh) with the default --home?  Just run
#  `install` here: wallet, key and settings are reused and the CPU service is replaced.
#
#  One-liner (on the machine):
#    curl -fsSL <raw-url>/proxima-gpu-miner.sh | bash -s -- install
# =============================================================================
set -Eeuo pipefail

SCRIPT_VERSION="1.1.0"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
DEFAULT_NODE_URL="${DEFAULT_NODE_URL:-http://161.97.92.49:8000}"
REPO_URL="${REPO_URL:-https://github.com/lunfardo314/proxima.git}"

# ---------- defaults ---------------------------------------------------------
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
MINER_HOME="${MINER_HOME:-$RUN_HOME/proxima-miner}"
BRANCH="${BRANCH:-master}"
BIN_DIR="${BIN_DIR:-}"                 # default: $MINER_HOME/bin  (no root needed for the binary)
NODE_URL="${NODE_URL:-}"
KEY_FILE="${KEY_FILE:-}"
GENERATE_KEY="${GENERATE_KEY:-0}"
MINER_WORKERS="${MINER_WORKERS:-}"
MINER_MAX_KHS="${MINER_MAX_KHS:-}"
NO_PUBLIC="${NO_PUBLIC:-}"
NO_SERVICES="${NO_SERVICES:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
KEY_MODE=""                            # import | generate | keep   (remembered in miner.conf)
SAVED_NODE_URL=""
USE_GPU="${USE_GPU:-}"                 # 1 = mine on the GPU(s), 0 = CPU only, empty = auto (use it if one is found)
GPU_DEVICES="${GPU_DEVICES:-}"         # CUDA device numbers, comma separated; empty = all
GPU_INTENSITY="${GPU_INTENSITY:-}"     # low | normal | high
GPU_ARCHS="${GPU_ARCHS:-}"             # e.g. "89 120"; empty = the installed GPUs
GPU_CCBIN="${GPU_CCBIN:-}"             # host compiler for nvcc (when the default gcc is too new)
WITH_SEEKER=0                          # 1 once the GPU seeker (Rust + CUDA) is part of this run
CUDA_REDIST_VERSION="${CUDA_REDIST_VERSION:-12.9.1}"   # private CUDA toolkit pieces (nvcc, cudart, cccl)
MIN_RUST="1.88"                        # the proxima crates need it

# public network nodes: independent witnesses for the ledger-library proof and extra
# mining-transaction streams (so that no single node can withhold a competitor's tx)
PUBLIC_APIS=("http://65.21.170.230:8001" "http://79.137.70.25:8001" "http://51.254.47.76:8001")
PUBLIC_NAMES=(hloc0 oseq1 oloc2)

MINER_SVC="proxima-gpu-miner"
OLD_CPU_SVC="proxima-cpu-miner"        # the CPU-only edition; replaced when installed with the same --home

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

# run as root (directly or through sudo)
as_root() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi; }
# run as the service user (drops root when invoked via sudo)
as_user() {
  if [[ $EUID -eq 0 && "$RUN_USER" != root ]]; then sudo -u "$RUN_USER" -H env "PATH=$PATH" "$@"
  else "$@"; fi
}
have_systemd() { [[ "$NO_SERVICES" != 1 ]] && command -v systemctl >/dev/null && [[ -d /run/systemd/system ]]; }

# ---------- prompts ----------------------------------------------------------
# Questions are read from the terminal even when the script itself is piped
# (curl ... | bash).  With -y, or with no terminal at all, the defaults are taken.
HAVE_TTY=0
if [[ "$ASSUME_YES" != 1 ]] && { : </dev/tty; } 2>/dev/null; then HAVE_TTY=1; fi

confirm() { # confirm "question" [default y|n]
  local def="${2:-n}" a=""
  if (( HAVE_TTY )); then read -r -p "  ${C_C}?${C_0} $1 [$([[ $def == y ]] && echo Y/n || echo y/N)] " a </dev/tty || a=""; fi
  a="${a:-$def}"; [[ "$a" =~ ^[Yy] ]]
}
ask() { # ask VAR "question" "default"
  local __v="$1" q="$2" d="${3:-}" a=""
  if (( HAVE_TTY )); then read -r -p "  ${C_C}?${C_0} $q${d:+ ${C_D}[$d]${C_0}}: " a </dev/tty || a=""; fi
  printf -v "$__v" '%s' "${a:-$d}"
}
ask_secret() { # ask_secret VAR "question"
  local __v="$1" a=""
  if (( HAVE_TTY )); then read -r -s -p "  ${C_C}?${C_0} $2: " a </dev/tty || a=""; echo >&2; fi
  printf -v "$__v" '%s' "$a"
}

# ---------- paths ------------------------------------------------------------
set_paths() {
  PROXIMA_HOME="$MINER_HOME"                 # name used by the shared toolchain helpers
  BIN_DIR="${BIN_DIR:-$MINER_HOME/bin}"
  SRC_DIR="$MINER_HOME/src/proxima"
  WALLET_DIR="$MINER_HOME/wallet"
  GO_ROOT="$MINER_HOME/.toolchain/go"
  GO_PATH="$MINER_HOME/.toolchain/gopath"
  BUILD_OUT="$MINER_HOME/build"
  BACKUP_DIR="$MINER_HOME/backups"
  CONF_FILE="$MINER_HOME/miner.conf"
  ENV_FILE="$MINER_HOME/miner.env"           # PROXIMA_KEY_PASSPHRASE of an encrypted key (mode 600)
  STAMP_FILE="$MINER_HOME/built.commit"      # source commit the installed proxi was built from
  PID_FILE="$MINER_HOME/miner.pid"
  LAUNCHER="$MINER_HOME/run-miner.sh"        # what systemd / the fallback actually runs
  LOG_FILE="$WALLET_DIR/miner.log"
  UNIT_FILE="/etc/systemd/system/$MINER_SVC.service"
  # GPU seeker: Rust + CUDA, all private to $MINER_HOME (nothing system-wide, no root)
  CUDA_ROOT="$MINER_HOME/.toolchain/cuda"
  CARGO_HOME_DIR="$MINER_HOME/.toolchain/cargo"
  RUSTUP_HOME_DIR="$MINER_HOME/.toolchain/rustup"
  GPU_SRC="$MINER_HOME/src/gpu-seeker"       # unpacked from the payload at the end of this script
  GPU_TARGET="$MINER_HOME/build/gpu-target"  # cargo target dir (kept: incremental rebuilds)
  GPU_BUILD="$MINER_HOME/build/gpu"
  GPU_INFO="$MINER_HOME/gpu.info"            # what the installed seeker was built from / for
  SEEKER_BIN="$BIN_DIR/gpu_seeker"
  SEEKER_FATBIN="$BIN_DIR/gpu_seeker.fatbin"
  SEEKER_PATH_DIR="$MINER_HOME/seeker-path"  # holds `nonce_seeker` -> gpu_seeker: where proxi looks for its seeker
  OLD_CPU_UNIT="/etc/systemd/system/$OLD_CPU_SVC.service"
}

# ---------- argument parsing -------------------------------------------------
CMD=""
ARGS=()
GPU_TUNE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-url)     NODE_URL="${2:?--node-url needs a value}"; shift 2;;
    --key-file)     KEY_FILE="${2:?--key-file needs a value}"; shift 2;;
    --generate-key) GENERATE_KEY=1; shift;;
    --workers)      MINER_WORKERS="${2:?--workers needs a value}"; shift 2;;
    --max-khs)      MINER_MAX_KHS="${2:?--max-khs needs a value}"; shift 2;;
    --gpu)          USE_GPU=1; shift;;
    --no-gpu)       USE_GPU=0; shift;;
    --gpu-devices)  GPU_DEVICES="${2:?--gpu-devices needs a value}"; GPU_TUNE=1; shift 2;;
    --gpu-intensity) GPU_INTENSITY="${2:?--gpu-intensity needs a value}"; GPU_TUNE=1; shift 2;;
    --gpu-arch)     GPU_ARCHS="${2:?--gpu-arch needs a value}"; shift 2;;
    --no-public)    NO_PUBLIC=1; shift;;
    --home)         MINER_HOME="${2:?--home needs a value}"; shift 2;;
    --branch)       BRANCH="${2:?--branch needs a value}"; shift 2;;
    --bin-dir)      BIN_DIR="${2:?--bin-dir needs a value}"; shift 2;;
    --no-services)  NO_SERVICES=1; shift;;
    --no-color)     COLOR=0; shift;;
    --color)        FORCE_COLOR=1; shift;;
    -y|--yes)       ASSUME_YES=1; HAVE_TTY=0; shift;;
    -h|--help)      CMD=help; shift;;
    -*)             die "unknown option: $1  (try: $0 help)";;
    *)              if [[ -z "$CMD" ]]; then CMD="$1"; else ARGS+=("$1"); fi; shift;;
  esac
done
CMD="${CMD:-help}"
set_paths
setup_colors

# ---------- settings that survive between runs (flags / env win over saved) --
norm_flags() { if [[ "${NO_PUBLIC:-0}" == 0 ]]; then NO_PUBLIC=""; fi; }
norm_flags
conf_get() { grep -E "^$1=" "$CONF_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true; }
load_conf() {
  [[ -f "$CONF_FILE" ]] || return 0
  [[ -n "$MINER_WORKERS" ]] || MINER_WORKERS=$(conf_get MINER_WORKERS)
  [[ -n "$MINER_MAX_KHS" ]] || MINER_MAX_KHS=$(conf_get MINER_MAX_KHS)
  [[ -n "$NO_PUBLIC"     ]] || NO_PUBLIC=$(conf_get NO_PUBLIC)
  [[ -n "$USE_GPU"       ]] || USE_GPU=$(conf_get USE_GPU)
  [[ -n "$GPU_DEVICES"   ]] || GPU_DEVICES=$(conf_get GPU_DEVICES)
  [[ -n "$GPU_INTENSITY" ]] || GPU_INTENSITY=$(conf_get GPU_INTENSITY)
  [[ -n "$GPU_ARCHS"     ]] || GPU_ARCHS=$(conf_get GPU_ARCHS)
  [[ -n "$KEY_MODE"      ]] || KEY_MODE=$(conf_get KEY_MODE)
  norm_flags
  return 0
}
save_conf() {
  as_user mkdir -p "$MINER_HOME"
  printf 'MINER_WORKERS=%s\nMINER_MAX_KHS=%s\nNO_PUBLIC=%s\nKEY_MODE=%s\nUSE_GPU=%s\nGPU_DEVICES=%s\nGPU_INTENSITY=%s\nGPU_ARCHS=%s\n' \
    "$MINER_WORKERS" "${MINER_MAX_KHS:-0}" "${NO_PUBLIC:-0}" "${KEY_MODE:-}" "${USE_GPU:-0}" "${GPU_DEVICES:-}" "${GPU_INTENSITY:-normal}" "${GPU_ARCHS:-}" | as_user tee "$CONF_FILE" >/dev/null
}
profile_node_url() { awk '/^[ \t]+(node_url|endpoint):/{print $2;exit}' "$WALLET_DIR/proxi.yaml" 2>/dev/null || true; }
profile_holder()   { awk '/holder_id:/{print $2;exit}' "$WALLET_DIR/proxi.yaml" 2>/dev/null || true; }
load_passphrase() { # an encrypted key needs its passphrase on every proxi call
  [[ -z "${PROXIMA_KEY_PASSPHRASE:-}" && -r "$ENV_FILE" ]] || return 0
  set -a; . "$ENV_FILE"; set +a
  return 0
}
load_install() { # for every command that works on an existing install
  load_conf
  [[ -f "$WALLET_DIR/proxi.yaml" ]] || die "nothing installed in $MINER_HOME — run: $0 install"
  NODE_URL=$(profile_node_url); [[ -n "$NODE_URL" ]] || die "no node_url in $WALLET_DIR/proxi.yaml"
  setup_ledger_apis
  load_passphrase
}

# nodes consulted for ledger info: the public ones first (an independent view), then your node
LEDGER_APIS=()
setup_ledger_apis() {
  LEDGER_APIS=()
  [[ -n "$NO_PUBLIC" ]] || LEDGER_APIS+=("${PUBLIC_APIS[@]}")
  [[ -z "$NODE_URL" ]] || LEDGER_APIS+=("$NODE_URL")
  return 0
}
net_get() { # net_get <api path> : first node that answers
  local u r
  for u in "${LEDGER_APIS[@]}"; do
    r=$(curl -fs -m 8 "$u$1" 2>/dev/null) && [[ -n "$r" ]] && { printf '%s' "$r"; return 0; }
  done
  return 1
}
sync_info() { curl -fs -m 5 "$1/api/v1/sync_info" 2>/dev/null || true; }
is_synced() { sync_info "$1" | grep -q '"synced": *true'; }

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

# =============================================================================
#  Your node: URL handling and probing
# =============================================================================
norm_url() { # 1.2.3.4 | 1.2.3.4:8000 | http://1.2.3.4:8000/  ->  http://1.2.3.4:8000
  local u="${1//[[:space:]]/}"
  u="${u%/}"
  [[ "$u" =~ ^https?:// ]] || u="http://$u"
  if ! [[ "${u#*//}" =~ :[0-9]+(/|$) ]] && ! [[ "$u" =~ ^https:// ]]; then u="$u:8000"; fi
  printf '%s' "$u"
}

NP_UP=0; NP_SYNCED=""; NP_CUR=""; NP_LRB=""; NP_RTT=""; NP_VER=""; NP_ID=""; NP_GEN=""
probe_node() { # probe_node <url>  -> NP_* ; returns 1 when the API does not answer
  local u="$1" out si ni lc
  NP_UP=0; NP_SYNCED=""; NP_CUR=""; NP_LRB=""; NP_RTT=""; NP_VER=""; NP_ID=""; NP_GEN=""
  out=$(curl -fs -m 8 -w '\n%{time_total}' "$u/api/v1/sync_info" 2>/dev/null) || return 1
  NP_RTT=$(tail -n1 <<<"$out" | awk '{printf "%d", $1 * 1000}')
  si=$(sed '$d' <<<"$out")
  jq -e . >/dev/null 2>&1 <<<"$si" || return 1
  NP_UP=1
  NP_SYNCED=$(jq -r '.synced // false' <<<"$si" 2>/dev/null || true)
  NP_CUR=$(jq -r '.current_slot // empty' <<<"$si" 2>/dev/null || true)
  NP_LRB=$(jq -r '.lrb_slot // empty' <<<"$si" 2>/dev/null || true)
  ni=$(curl -fs -m 6 "$u/api/v1/node_info" 2>/dev/null || true)
  NP_VER=$(jq -r '.version // empty' <<<"$ni" 2>/dev/null || true)
  NP_ID=$(jq -r '.id // empty' <<<"$ni" 2>/dev/null || true)
  lc=$(curl -fs -m 6 "$u/api/v1/ledger_constants" 2>/dev/null || true)
  NP_GEN=$(jq -r '.genesis_time_unix // empty' <<<"$lc" 2>/dev/null || true)
  return 0
}

# genesis time of the public network ("" when no public node answers, or with --no-public)
public_genesis() {
  local u g
  [[ -z "$NO_PUBLIC" ]] || return 0
  for u in "${PUBLIC_APIS[@]}"; do
    g=$(curl -fs -m 6 "$u/api/v1/ledger_constants" 2>/dev/null | jq -r '.genesis_time_unix // empty' 2>/dev/null || true)
    if [[ -n "$g" ]]; then printf '%s' "$g"; return 0; fi
  done
  return 0
}

# choose_node: sets NODE_URL (asks unless given); validates reachability, sync and ledger
choose_node() {
  step "Your node"
  local cur="${NODE_URL:-${SAVED_NODE_URL:-$DEFAULT_NODE_URL}}" url try pg
  for try in 1 2 3; do
    if [[ -z "$NODE_URL" || $try -gt 1 ]]; then ask url "API URL of your synced node" "$cur"; else url="$NODE_URL"; fi
    url=$(norm_url "$url")
    log "checking $url …"
    if probe_node "$url"; then
      ok "node API reachable — ${NP_RTT} ms, ${NP_VER:-?}, id ${NP_ID:0:16}…"
      if [[ "$NP_SYNCED" == true ]]; then ok "node is SYNCED (slot ${NP_CUR:-?}, LRB ${NP_LRB:-?})"
      else warn "node reports NOT synced (slot ${NP_CUR:-?}, LRB ${NP_LRB:-?}) — the miner will wait for it"; fi
      pg=$(public_genesis)
      if [[ -n "$pg" && -n "$NP_GEN" && "$pg" != "$NP_GEN" ]]; then
        warn "this node's ledger (genesis $NP_GEN) differs from the public network ($pg):"
        warn "it is probably on an OLD ledger — run 'proxima-setup.sh reset' on the node first."
        confirm "use it anyway?" n || { NODE_URL=""; cur="$url"; continue; }
      fi
      NODE_URL="$url"; return 0
    fi
    warn "no answer from $url/api/v1/sync_info"
    warn "on the node: the API must listen on 0.0.0.0 and its firewall must let THIS vps reach the API port (TCP)"
    if (( ! HAVE_TTY )); then die "node not reachable: $url"; fi
    cur="$url"; NODE_URL=""
  done
  die "no reachable node given"
}

# =============================================================================
#  Pieces shared with proxima-setup.sh (same code, same behaviour)
# =============================================================================
install_deps() {
  step "System dependencies"
  local need=()
  for c in git curl tar jq; do command -v "$c" >/dev/null || need+=("$c"); done
  # Go binaries are built with CGO_ENABLED=0 (static); a C/C++ compiler is needed for Rust (linker) and nvcc (host side)
  if [[ "$WITH_SEEKER" == 1 ]]; then
    command -v cc  >/dev/null || need+=(gcc)
    if command -v apt-get >/dev/null; then
      command -v g++ >/dev/null || need+=(g++)
      command -v xz  >/dev/null || need+=(xz-utils)
    else
      command -v g++ >/dev/null || need+=(gcc-c++)
      command -v xz  >/dev/null || need+=(xz)
    fi
  fi
  if [[ ${#need[@]} -eq 0 ]]; then ok "git, curl, tar, jq$([[ "$WITH_SEEKER" == 1 ]] && echo ', gcc/g++, xz') present"; return; fi
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
low_mem() { # < 3 GB RAM: the go compiler can be OOM-killed with full parallelism
  local kb; kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  (( ${kb:-0} > 0 && ${kb:-0} < 3000000 ))
}
src_miner_version() { # miner version this source tree writes into transits
  grep -hoE 'MinerVersion +uint32 += +[0-9]+' "$SRC_DIR/proxi/node_cmd/mine/mine.go" 2>/dev/null | grep -oE '[0-9]+$' | head -1 || true
}
minor_of() { echo "$1" | sed -E 's/^v?([0-9]+\.[0-9]+).*/\1/'; }
check_compat() {
  load_net_info || { warn "could not read ledger info from the public nodes (offline?) — skipping compatibility checks"; return 0; }
  log "network ledger: ${C_B}${NET_DESC:-?}${C_0}  genesis $(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS") UTC  nodes ${NET_NODE_VER:-?}"
  if [[ -n "${SRC_VERSION:-}" && -n "$NET_NODE_VER" && "$(minor_of "$SRC_VERSION")" != "$(minor_of "$NET_NODE_VER")" ]]; then
    warn "source is $SRC_VERSION but the network runs $NET_NODE_VER — a different minor version is usually incompatible."
    warn "pin a matching tag or commit with --branch <tag|commit> (v1.0.1-take1 = commit f5370038)."
  fi
  local mv; mv=$(src_miner_version)
  if [[ -n "$mv" && -n "$NET_MINER_VER" && "$mv" != "$NET_MINER_VER" ]]; then
    warn "miner version in source is $mv, the ledger requires $NET_MINER_VER — mined transits would be invalid."
  fi
}
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
public_ip() { local u ip
  for u in https://api.ipify.org https://ifconfig.me/ip https://icanhazip.com; do
    ip=$(curl -4fs -m 4 "$u" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then echo "$ip"; return 0; fi
  done; return 0; }

# the file this script runs from ("" when piped through curl | bash)
SELF_PATH=""
if [[ -f "${BASH_SOURCE[0]:-}" ]]; then SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"; fi
SELF_URLS=("https://raw.githubusercontent.com/daddyxminer-cell/my/main/proxima-gpu-miner.sh"
           "https://raw.githubusercontent.com/daddyxminer-cell/my/genspark_ai_developer/proxima-gpu-miner.sh")

KEY_TMP=""
cleanup() { if [[ -n "$KEY_TMP" && -f "$KEY_TMP" ]]; then rm -f "$KEY_TMP" 2>/dev/null || as_root rm -f "$KEY_TMP" || true; fi; }
trap cleanup EXIT

# =============================================================================
#  Preflight
# =============================================================================
ROOTLESS=0
preflight() {
  [[ "$(uname -s)" == Linux ]] || die "Linux only"
  as_user mkdir -p "$MINER_HOME"
  local free; free=$(df -Pk "$MINER_HOME" | awk 'NR==2{print int($4/1024/1024)}')
  local need_gb=3; [[ "$WITH_SEEKER" != 1 ]] || need_gb=6   # Go + proxi build; plus CUDA compiler, Rust, crates, kernel
  (( free >= need_gb )) || die "only ${free} GB free in $MINER_HOME (need >= $need_gb GB for the toolchains + build)"
  local kb; kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  if (( ${kb:-0} > 0 && ${kb:-0} < 1500000 )); then
    warn "only $((kb / 1024)) MB RAM: compiling proxi may be OOM-killed — if it fails, add swap:"
    warn "  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
  fi
  # root is only needed for packages and the systemd unit
  if [[ $EUID -ne 0 ]]; then
    local have_sudo=0
    if command -v sudo >/dev/null; then
      if sudo -n true 2>/dev/null; then have_sudo=1
      elif (( HAVE_TTY )); then log "sudo is needed for packages and the systemd service — you may be asked for your password"
        if sudo -v; then have_sudo=1; fi
      fi
    fi
    if (( ! have_sudo )); then
      warn "no root / sudo: no packages will be installed and no systemd service created (background process instead)"
      ROOTLESS=1; NO_SERVICES=1
    fi
  fi
}

ensure_deps() {
  local missing=() c
  for c in git curl tar jq; do command -v "$c" >/dev/null || missing+=("$c"); done
  if [[ "$WITH_SEEKER" == 1 ]]; then for c in cc g++ xz; do command -v "$c" >/dev/null || missing+=("$c"); done; fi
  if (( ROOTLESS && ${#missing[@]} )); then die "missing tools and no root to install them: ${missing[*]}"; fi
  install_deps
}

# =============================================================================
#  Build: `proxi` (wallet + miner). No node, no database. (Rust + CUDA only for the GPU seeker.)
# =============================================================================
BUILT_COMMIT=""
build_proxi() {
  step "Compiling proxi $SRC_VERSION  (miner + wallet)"
  as_user mkdir -p "$BUILD_OUT"
  local par=()
  if low_mem; then par=(-p 1); warn "less than 3 GB RAM: building with -p 1 (slower, avoids OOM)"; fi
  ( cd "$SRC_DIR"
    log "proxi …  (the first build downloads Go modules — a few minutes)"
    go_run build ${par[@]+"${par[@]}"} -trimpath -o "$BUILD_OUT/proxi.new" ./proxi )
  local v; v=$("$BUILD_OUT/proxi.new" version 2>/dev/null | awk '/Version:/{print $2}' || true)
  [[ "$v" == "$SRC_VERSION" ]] || die "built proxi reports '${v:-nothing}', expected $SRC_VERSION"
  BUILT_COMMIT=$(as_user git -C "$SRC_DIR" rev-parse HEAD)
  ok "proxi $v (commit ${BUILT_COMMIT:0:10})"
}

installed_version() { [[ -x "$BIN_DIR/proxi" ]] && "$BIN_DIR/proxi" version 2>/dev/null | awk '/Version:/{print $2}' || true; }

install_self() {
  # keep a copy of this script next to the install, so 'status' & co. work after a curl | bash
  local self="$MINER_HOME/proxima-gpu-miner.sh" u
  if [[ -n "$SELF_PATH" ]]; then
    [[ "$SELF_PATH" == "$self" ]] || as_user install -m 0755 "$SELF_PATH" "$self"
  else
    for u in "${SELF_URLS[@]}"; do
      if as_user curl -fsSL -m 20 "$u" -o "$self.tmp" 2>/dev/null && head -1 "$self.tmp" | grep -q '^#!'; then
        as_user chmod 755 "$self.tmp"; as_user mv -f "$self.tmp" "$self"; break; fi
      as_user rm -f "$self.tmp"
    done
  fi
  if [[ -x "$self" ]]; then
    ok "script -> $self"
    if [[ ! -e /usr/local/bin/proxima-gpu-miner ]] && { [[ $EUID -eq 0 ]] || sudo -n true 2>/dev/null; }; then
      as_root ln -sf "$self" /usr/local/bin/proxima-gpu-miner 2>/dev/null && ok "shortcut: proxima-gpu-miner <command>" || true
    fi
  else warn "could not keep a copy of this script — save it as $self to use 'status'/'watch' later"; fi
}

install_binary() {
  step "Installing proxi to $BIN_DIR"
  local dst="$BIN_DIR/proxi"
  as_user mkdir -p "$BIN_DIR" 2>/dev/null || as_root mkdir -p "$BIN_DIR"
  if [[ -f "$dst" ]]; then { as_user cp -f "$dst" "$dst.prev" 2>/dev/null || as_root cp -f "$dst" "$dst.prev"; } || true; fi
  { as_user install -m 0755 "$BUILD_OUT/proxi.new" "$dst.tmp" 2>/dev/null || as_root install -m 0755 "$BUILD_OUT/proxi.new" "$dst.tmp"; }
  { as_user mv -f "$dst.tmp" "$dst" 2>/dev/null || as_root mv -f "$dst.tmp" "$dst"; }
  rm -f "$BUILD_OUT/proxi.new"
  [[ -z "$BUILT_COMMIT" ]] || printf '%s\n' "$BUILT_COMMIT" | as_user tee "$STAMP_FILE" >/dev/null
  ok "proxi -> $dst"
  install_self
  hash -r
}

# =============================================================================
#  Wallet key: import yours (rewards land in YOUR wallet) or generate one
# =============================================================================
key_json_ok()   { jq -e '(.version != null) and ((.private_key // .crypto) != null)' "$1" >/dev/null 2>&1; }
key_encrypted() { jq -e '.crypto != null' "$1" >/dev/null 2>&1; }
key_holder_of() { jq -r '.holder_id // empty' "$1" 2>/dev/null || true; }
installed_key() { [[ -f "$WALLET_DIR/proxima.key" && -f "$WALLET_DIR/proxi.yaml" ]]; }
expand_path()   { local p="$1"; p="${p/#\~/$RUN_HOME}"; printf '%s' "$p"; }

# read a pasted JSON key from the terminal (no echo): lines are collected until they form valid JSON
paste_key() { # -> KEY_TMP
  local buf="" line n=0
  as_user mkdir -p "$MINER_HOME"
  log "paste the content of proxima.key now (nothing is echoed) — it ends by itself once the JSON is complete"
  while (( n < 60 )); do
    IFS= read -r -s line </dev/tty || break
    n=$((n + 1)); buf+="$line"$'\n'; printf '.' >&2
    if jq -e . >/dev/null 2>&1 <<<"$buf"; then break; fi
  done
  echo >&2
  jq -e . >/dev/null 2>&1 <<<"$buf" || die "that is not valid JSON (got $n lines)"
  KEY_TMP=$(as_user mktemp -p "$MINER_HOME" key.XXXXXX)
  printf '%s' "$buf" | as_user tee "$KEY_TMP" >/dev/null
  KEY_FILE="$KEY_TMP"
}

# choose_key: decide KEY_MODE (import | generate | keep) and KEY_FILE; asks unless flags decide
choose_key() {
  step "Wallet key — who receives the rewards"
  if [[ -n "$KEY_FILE" ]]; then
    KEY_FILE=$(expand_path "$KEY_FILE"); [[ -f "$KEY_FILE" ]] || die "key file not found: $KEY_FILE"
    KEY_MODE=import
  elif [[ "$GENERATE_KEY" == 1 ]]; then KEY_MODE=generate
  elif (( HAVE_TTY )); then
    cat <<EOT
  Mining is non-outsourceable: the reward is paid to the wallet of the key that
  signs the mining transaction. So:
    ${C_B}import${C_0}   your main wallet's ${C_W}proxima.key${C_0}   -> rewards go straight to your main wallet
    ${C_B}generate${C_0} a new key on this VPS            -> rewards go to a new wallet (sweep it later)
  ${C_D}(on your node the key is usually ~/proxima/wallet/proxima.key)${C_0}

EOT
    local def=1 c
    printf '    1) import an existing key  (give a path, or paste the JSON)\n    2) generate a new key\n'
    if installed_key; then printf '    3) keep the key installed here  (holder %s…)\n' "$(profile_holder | cut -c1-16)"; def=3; fi
    for _ in 1 2 3; do
      ask c "choose" "$def"
      case "$c" in
        1|import)   KEY_MODE=import; break;;
        2|generate) KEY_MODE=generate; break;;
        3|keep)     if installed_key; then KEY_MODE=keep; break; fi; warn "no key installed here";;
        *)          warn "enter 1, 2${def:+ or 3}";;
      esac
    done
    [[ -n "$KEY_MODE" ]] || die "no key chosen"
    if [[ "$KEY_MODE" == import ]]; then
      local p
      ask p "path to proxima.key  (empty = paste the JSON here)" ""
      if [[ -n "$p" ]]; then KEY_FILE=$(expand_path "$p"); [[ -f "$KEY_FILE" ]] || die "file not found: $KEY_FILE"
      else paste_key; fi
    fi
  else
    if installed_key; then KEY_MODE=keep
    else KEY_MODE=generate
      warn "no --key-file given and no terminal to ask: generating a NEW key — rewards go to a new wallet"
    fi
  fi
  if [[ "$KEY_MODE" == import ]]; then
    key_json_ok "$KEY_FILE" || die "$KEY_FILE does not look like a Proxima key file (expected the JSON written by proxi)"
    ok "key to import: $(key_encrypted "$KEY_FILE" && echo 'encrypted' || echo 'plain') keystore"
  fi
}

backup_key() {
  [[ -f "$WALLET_DIR/proxima.key" ]] || return 0
  as_user mkdir -p "$BACKUP_DIR"; as_user chmod 700 "$BACKUP_DIR"
  local f="$BACKUP_DIR/wallet-$(date +%Y%m%d-%H%M%S).tar.gz" files=(proxima.key)
  [[ -f "$WALLET_DIR/proxi.yaml" ]] && files+=(proxi.yaml)
  as_user tar -C "$WALLET_DIR" -czf "$f" "${files[@]}"
  as_user chmod 600 "$f"
  ok "wallet backup: $f  ${C_Y}(copy it somewhere safe!)${C_0}"
}

# an encrypted key needs its passphrase on every start: verify it, keep it (mode 600) for the launcher
store_passphrase() { # store_passphrase <key file>
  local f="$1" p="${PROXIMA_KEY_PASSPHRASE:-}" try
  for try in 1 2 3; do
    if [[ -z "$p" ]]; then
      (( HAVE_TTY )) || die "the key is encrypted: set PROXIMA_KEY_PASSPHRASE (no terminal to ask)"
      ask_secret p "passphrase of the key"
    fi
    if PROXIMA_KEY_PASSPHRASE="$p" "$BIN_DIR/proxi" util key info --file "$f" --verify >/dev/null 2>&1; then
      printf 'PROXIMA_KEY_PASSPHRASE=%q\n' "$p" | as_user tee "$ENV_FILE" >/dev/null
      as_user chmod 600 "$ENV_FILE"
      PROXIMA_KEY_PASSPHRASE="$p"; export PROXIMA_KEY_PASSPHRASE
      ok "passphrase verified and stored in $ENV_FILE (mode 600)"
      return 0
    fi
    warn "wrong passphrase"; p=""
  done
  die "could not unlock the key"
}

# bring proxi.yaml in line with the chosen node (and --no-public)
profile_apply() { # profile_apply <node url>
  local y="$WALLET_DIR/proxi.yaml" url="$1"
  as_user sed -i -E "s#^([[:space:]]*node_url:).*#\1 $url#" "$y"
  if [[ -n "$NO_PUBLIC" ]]; then
    awk '/^[ \t]*node_urls:/{ match($0,/^[ \t]*/); print substr($0,1,RLENGTH) "node_urls: []"; skip=1; next }
         skip && /^[ \t]+-[ \t]/ {next}  {skip=0; print}' "$y" | as_user tee "$y.new" >/dev/null
    as_user mv -f "$y.new" "$y"; as_user chmod 600 "$y"
    warn "--no-public: no witness nodes in the profile — the node's ledger library is NOT cross-checked"
  else
    migrate_wallet_profile
  fi
}

# create / replace the wallet profile according to KEY_MODE
apply_key() {
  step "Wallet ($WALLET_DIR)"
  as_user mkdir -p "$WALLET_DIR"; as_user chmod 700 "$WALLET_DIR"
  local holder
  case "$KEY_MODE" in
    keep)
      installed_key || die "no key installed here to keep"
      ok "keeping the installed key, holder $(profile_holder)";;
    import|generate)
      # 1) validate EVERYTHING first: a bad file or passphrase must leave the installed wallet untouched
      if [[ "$KEY_MODE" == import ]]; then
        local out
        out=$(PROXIMA_KEY_PASSPHRASE="${PROXIMA_KEY_PASSPHRASE:-}" "$BIN_DIR/proxi" util key info --file "$KEY_FILE" 2>&1 || true)
        grep -q 'Public key:' <<<"$out" || die "proxi cannot read $KEY_FILE: $(tail -n1 <<<"$out")"
        if key_encrypted "$KEY_FILE"; then store_passphrase "$KEY_FILE"
        else
          PROXIMA_KEY_PASSPHRASE="" "$BIN_DIR/proxi" util key info --file "$KEY_FILE" --verify >/dev/null 2>&1 \
            || die "the key file is damaged: private and public key do not match"
          as_user rm -f "$ENV_FILE"
        fi
      else
        as_user rm -f "$ENV_FILE"
      fi
      # 2) only now replace the wallet (the old one is backed up first)
      backup_key
      as_user rm -f "$WALLET_DIR/proxi.yaml" "$WALLET_DIR/proxima.key"
      if [[ "$KEY_MODE" == import ]]; then
        as_user install -m 600 "$KEY_FILE" "$WALLET_DIR/proxima.key"
        # 'config wallet' finds proxima.key and asks "Use it?" -> y
        ( cd "$WALLET_DIR" && printf 'y\n' | as_user "$BIN_DIR/proxi" config wallet >/dev/null )
        ok "imported key, holder $(profile_holder)"
      else
        # new key: 'config wallet' asks "Encrypt?" -> n
        ( cd "$WALLET_DIR" && printf 'n\n' | as_user "$BIN_DIR/proxi" config wallet >/dev/null )
        ok "generated a NEW key, holder $(profile_holder)"
        warn "rewards go to THIS new wallet. Back up $WALLET_DIR/proxima.key, then sweep to your main wallet with:"
        warn "  cd $WALLET_DIR && $BIN_DIR/proxi node send_to_wallet <amount> <your main holder ID>"
      fi
      as_user chmod 600 "$WALLET_DIR/proxima.key";;
  esac
  [[ -f "$WALLET_DIR/proxi.yaml" ]] || die "wallet profile was not created"
  [[ -z "$(profile_holder)" ]] && die "no holder_id in $WALLET_DIR/proxi.yaml"
  profile_apply "$NODE_URL"
  ok "miner/wallet node API: $NODE_URL"
}

# =============================================================================
#  GPU: detection, CUDA compiler, Rust toolchain, building + installing the seeker
# =============================================================================
GPU_N=0; GPU_IDX=(); GPU_NAMES=(); GPU_CCS=(); GPU_MEMS=(); GPU_DRIVER=""; GPU_PCI_NAME=""
NVCC=""

trim() { local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
ver_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]; }   # ver_ge have want

cc_from_name() { # compute capability from the marketing name, for drivers that cannot say
  case "$1" in
    *"RTX 50"[0-9][0-9]*|*"RTX PRO"*"Blackwell"*) echo 12.0;;
    *"RTX 40"[0-9][0-9]*|*"RTX 6000 Ada"*|*"RTX 5000 Ada"*|*"RTX 4500 Ada"*|*"RTX 4000 Ada"*|*"RTX 2000 Ada"*|*"L40"*|*"L4"|*"L4 "*) echo 8.9;;
    *"RTX 30"[0-9][0-9]*|*"RTX A"[0-9]*|*"A40"*|*"A10"*|*"A16"*) echo 8.6;;
    *"RTX 20"[0-9][0-9]*|*"GTX 16"[0-9][0-9]*|*"T4"*) echo 7.5;;
    *) echo "";;
  esac
}

# fills GPU_* from nvidia-smi (lspci as a fallback to tell "no GPU" from "no driver")
detect_gpus() {
  GPU_N=0; GPU_IDX=(); GPU_NAMES=(); GPU_CCS=(); GPU_MEMS=(); GPU_DRIVER=""; GPU_PCI_NAME=""
  local out="" line idx name cc drv mem
  if command -v nvidia-smi >/dev/null 2>&1; then
    out=$(timeout 25 nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null) || out=""
    if [[ -z "$out" ]]; then   # an old driver has no compute_cap field
      out=$(timeout 25 nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '{print $1", "$2", , "$3", "$4}') || out=""
    fi
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      IFS=',' read -r idx name cc drv mem <<<"$line"
      idx=$(trim "$idx"); name=$(trim "$name"); cc=$(trim "$cc"); drv=$(trim "$drv"); mem=$(trim "$mem")
      [[ "$idx" =~ ^[0-9]+$ ]] || continue
      [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]] || cc=$(cc_from_name "$name")
      GPU_IDX+=("$idx"); GPU_NAMES+=("$name"); GPU_CCS+=("$cc"); GPU_MEMS+=("${mem:-0}")
      GPU_DRIVER="$drv"; GPU_N=$((GPU_N + 1))
    done <<<"$out"
  fi
  if (( GPU_N == 0 )) && command -v lspci >/dev/null 2>&1; then
    GPU_PCI_NAME=$(lspci 2>/dev/null | grep -iE 'nvidia' | grep -iE 'vga|3d|display' | head -n1 | sed -E 's/^[^:]+:[^:]+: *//' || true)
  fi
  return 0
}

arch_of_cc() { printf '%s%s' "${1%%.*}" "${1##*.}"; }   # 8.9 -> 89, 12.0 -> 120
gpu_arch_list() { # CUDA architectures to compile for, one line, e.g. "89 120"
  local cc a list=()
  if [[ -n "$GPU_ARCHS" ]]; then tr ',' ' ' <<<"$GPU_ARCHS" | xargs; return 0; fi
  for cc in ${GPU_CCS[@]+"${GPU_CCS[@]}"}; do [[ -n "$cc" ]] && list+=("$(arch_of_cc "$cc")"); done
  if (( ${#list[@]} == 0 )); then echo "89 120"; return 0; fi
  for a in "${list[@]}"; do echo "$a"; done | sort -un | xargs
}
check_archs() {
  local a
  for a in $(gpu_arch_list); do
    [[ "$a" =~ ^[0-9]{2,3}$ ]] && (( a >= 75 )) || die "bad CUDA architecture '$a' (--gpu-arch \"89 120\"; the kernel needs compute capability >= 7.5)"
  done
}
need_cuda_ver() { local a max=0; for a in $(gpu_arch_list); do (( a > max )) && max=$a; done; if (( max >= 100 )); then echo 12.8; else echo 12.0; fi; }

gpu_params() { # GPU_INTENSITY -> kernel length / queue depth (the seeker adapts the real launch size to the target)
  case "${GPU_INTENSITY:-normal}" in
    low)    GPU_TARGET_MS=30;  GPU_BPS=2;;
    high)   GPU_TARGET_MS=300; GPU_BPS=8;;
    *)      GPU_TARGET_MS=150; GPU_BPS=4;;
  esac
}

nvcc_version() { "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1 || true; }
find_nvcc() { # find_nvcc <min version>: first nvcc that is new enough (private copy, PATH, /usr/local/cuda*)
  local minv="$1" c v d
  local cands=("$CUDA_ROOT/bin/nvcc")
  if command -v nvcc >/dev/null 2>&1; then cands+=("$(command -v nvcc)"); fi
  for d in /usr/local/cuda /usr/local/cuda-*; do [[ -x "$d/bin/nvcc" ]] && cands+=("$d/bin/nvcc"); done
  for c in "${cands[@]}"; do
    [[ -x "$c" ]] || continue
    v=$(nvcc_version "$c"); [[ -n "$v" ]] || continue
    if ver_ge "$v" "$minv"; then printf '%s' "$c"; return 0; fi
  done
  return 1
}
host_ccbin() { # nvcc 12.x supports gcc <= 14: pick an older g++ when the default is newer (or use --gpu-ccbin via GPU_CCBIN)
  if [[ -n "$GPU_CCBIN" ]]; then printf '%s' "$GPU_CCBIN"; return 0; fi
  local v n p; v=$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || true)
  if [[ "$v" =~ ^[0-9]+$ ]] && (( v > 14 )); then
    for n in 14 13 12; do p=$(command -v "g++-$n" 2>/dev/null || true); if [[ -n "$p" ]]; then printf '%s' "$p"; return 0; fi; done
  fi
  return 0
}

ensure_cuda() {
  step "CUDA compiler (nvcc)"
  check_archs
  local need; need=$(need_cuda_ver)
  if NVCC=$(find_nvcc "$need"); then ok "nvcc $(nvcc_version "$NVCC") — $NVCC"; return 0; fi
  local plat base="https://developer.download.nvidia.com/compute/cuda/redist"
  case "$(uname -m)" in x86_64|amd64) plat=linux-x86_64;; aarch64|arm64) plat=linux-sbsa;; *) die "no CUDA compiler for $(uname -m)";; esac
  command -v xz >/dev/null || die "xz is needed to unpack the CUDA compiler (apt-get install -y xz-utils)"
  log "no CUDA compiler >= $need on this machine — downloading CUDA $CUDA_REDIST_VERSION nvcc + cudart + cccl (~85 MB, private to $CUDA_ROOT, nothing system-wide)"
  local json tmp c rel sum
  json=$(curl -fsSL -m 60 "$base/redistrib_$CUDA_REDIST_VERSION.json") || die "cannot download $base/redistrib_$CUDA_REDIST_VERSION.json"
  tmp=$(as_user mktemp -d -p "$MINER_HOME")
  as_user rm -rf "$CUDA_ROOT.new"; as_user mkdir -p "$CUDA_ROOT.new"
  for c in cuda_nvcc cuda_cudart cuda_cccl; do
    rel=$(jq -r --arg c "$c" --arg p "$plat" '.[$c][$p].relative_path // empty' <<<"$json")
    sum=$(jq -r --arg c "$c" --arg p "$plat" '.[$c][$p].sha256 // empty' <<<"$json")
    [[ -n "$rel" && -n "$sum" ]] || die "CUDA $CUDA_REDIST_VERSION has no $c for $plat (set CUDA_REDIST_VERSION to another release)"
    log "  $c …"
    as_user curl -fsSL "$base/$rel" -o "$tmp/$c.tar.xz" || die "download failed: $base/$rel"
    echo "$sum  $tmp/$c.tar.xz" | sha256sum -c --quiet - || die "checksum mismatch for $c — refusing to use it"
    as_user tar -xJf "$tmp/$c.tar.xz" -C "$CUDA_ROOT.new" --strip-components=1
  done
  as_user rm -rf "$CUDA_ROOT" "$tmp"; as_user mv "$CUDA_ROOT.new" "$CUDA_ROOT"
  NVCC="$CUDA_ROOT/bin/nvcc"
  local v; v=$(nvcc_version "$NVCC"); [[ -n "$v" ]] || die "the downloaded nvcc does not run on this machine"
  ok "nvcc $v — $NVCC  (checksums verified)"
}

rust_env() { as_user env PATH="$CARGO_HOME_DIR/bin:$PATH" CARGO_HOME="$CARGO_HOME_DIR" RUSTUP_HOME="$RUSTUP_HOME_DIR" "$@"; }
rust_ok() { local v; v=$(rust_env rustc --version 2>/dev/null | awk '{print $2}' || true); [[ -n "$v" ]] && ver_ge "$v" "$MIN_RUST"; }
ensure_rust() {
  step "Rust toolchain (for the GPU seeker)"
  as_user mkdir -p "$MINER_HOME/.toolchain"
  if rust_ok; then ok "rustc $(rust_env rustc --version | awk '{print $2}')"; return 0; fi
  if [[ -x "$CARGO_HOME_DIR/bin/rustup" ]]; then
    log "updating the private Rust toolchain (the proxima crates need rustc >= $MIN_RUST)"
    rust_env rustup update stable --no-self-update >/dev/null 2>&1 || true
  else
    log "installing the Rust toolchain (rustup, private to $MINER_HOME/.toolchain; needs no root)"
    as_user env CARGO_HOME="$CARGO_HOME_DIR" RUSTUP_HOME="$RUSTUP_HOME_DIR" \
      sh -c 'curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/dev/null 2>&1' \
      || die "could not install Rust (is https://sh.rustup.rs reachable?)"
  fi
  rust_ok || die "no Rust >= $MIN_RUST available after installing"
  ok "rustc $(rust_env rustc --version | awk '{print $2}')"
}

gpu_payload_hash() { gpu_payload | sha256sum | cut -c1-16; }
unpack_gpu_src() {
  as_user rm -rf "$GPU_SRC"; as_user mkdir -p "$GPU_SRC"
  gpu_payload | base64 -d | as_user tar -xz -C "$GPU_SRC"
  sed "s#@PROXIMA_SRC@#$SRC_DIR#" "$GPU_SRC/Cargo.toml.in" | as_user tee "$GPU_SRC/Cargo.toml" >/dev/null
}
gpu_build_key() { # what the installed seeker was built from: this script's sources, the proxima commit, the architectures
  printf 'payload=%s proxima=%s archs=%s' "$(gpu_payload_hash)" "$(as_user git -C "$SRC_DIR" rev-parse --short=10 HEAD 2>/dev/null || echo ?)" "$(gpu_arch_list)"
}
gpu_stale() { # 0 = the seeker has to be (re)built
  [[ -x "$SEEKER_BIN" && -f "$SEEKER_FATBIN" && -f "$GPU_INFO" ]] || return 0
  [[ "$(head -n1 "$GPU_INFO" 2>/dev/null)" == "$(gpu_build_key)" ]] && return 1
  return 0
}

build_gpu_seeker() {
  step "GPU seeker: CUDA kernel + Rust host"
  [[ -d "$SRC_DIR/util/vrf/rust" ]] || die "proxima source not found in $SRC_DIR (fetch_source first)"
  check_archs
  unpack_gpu_src
  as_user mkdir -p "$GPU_BUILD" "$GPU_TARGET"
  local archs a max=0 gen=() ccb; archs=$(gpu_arch_list)
  for a in $archs; do gen+=(-gencode "arch=compute_$a,code=sm_$a"); (( a > max )) && max=$a; done
  gen+=(-gencode "arch=compute_$max,code=compute_$max")      # PTX: newer cards JIT it
  ccb=$(host_ccbin)
  local kb; kb=$(awk '/^(MemAvailable|SwapFree):/{s+=$2} END{print s+0}' /proc/meminfo 2>/dev/null || echo 0)
  if (( kb > 0 && kb < 1700000 )); then
    warn "only $((kb / 1024)) MB of free RAM+swap: nvcc needs about 1 GB for the Blackwell (sm_120) kernel and may be OOM-killed — add swap if it fails:"
    warn "  fallocate -l 3G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
  fi
  log "nvcc: sm_${archs// /, sm_} + PTX compute_$max${ccb:+  (host compiler $ccb)}  — 1 to 4 minutes"
  if ! ( cd "$GPU_SRC/cuda" && as_user env PATH="$(dirname "$NVCC"):$PATH" "$NVCC" -U_GNU_SOURCE -O3 -std=c++14 --fatbin \
           "${gen[@]}" ${ccb:+-ccbin "$ccb"} -o "$GPU_BUILD/gpu_seeker.fatbin.new" kernel.cu ); then
    die "nvcc failed. Typical causes: a host gcc newer than 14 (install g++-13 or pass GPU_CCBIN=/usr/bin/g++-13), out of memory (add swap), or an architecture this nvcc does not know (--gpu-arch \"89\")"
  fi
  ok "kernel compiled ($(du -k "$GPU_BUILD/gpu_seeker.fatbin.new" | cut -f1) KB)"
  log "Rust host … (the first build downloads crates — a few minutes)"
  local jobs=(); if low_mem; then jobs=(-j 1); fi
  ( cd "$GPU_SRC" && rust_env env CARGO_TARGET_DIR="$GPU_TARGET" RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet ${jobs[@]+"${jobs[@]}"} ) \
    || die "building the Rust GPU seeker failed (scroll up for the compiler message)"
  as_user cp -f "$GPU_TARGET/release/gpu_seeker" "$GPU_BUILD/gpu_seeker.new"
  ok "gpu_seeker built"
}

put_file() { # put_file <src> <dst> <mode>   (as the user, or as root for a root-owned --bin-dir)
  local src="$1" dst="$2" mode="$3"
  { as_user install -m "$mode" "$src" "$dst.tmp" 2>/dev/null || as_root install -m "$mode" "$src" "$dst.tmp"; }
  { as_user mv -f "$dst.tmp" "$dst" 2>/dev/null || as_root mv -f "$dst.tmp" "$dst"; }
}
install_gpu_files() {
  step "Installing the GPU seeker to $BIN_DIR"
  as_user mkdir -p "$BIN_DIR" 2>/dev/null || as_root mkdir -p "$BIN_DIR"
  put_file "$GPU_BUILD/gpu_seeker.new" "$SEEKER_BIN" 0755
  put_file "$GPU_BUILD/gpu_seeker.fatbin.new" "$SEEKER_FATBIN" 0644
  as_user rm -f "$GPU_BUILD/gpu_seeker.new" "$GPU_BUILD/gpu_seeker.fatbin.new"
  # `proxi node mine --seeker` looks for a program called nonce_seeker on PATH: the launcher puts this directory first
  as_user mkdir -p "$SEEKER_PATH_DIR"; as_user ln -sf "$SEEKER_BIN" "$SEEKER_PATH_DIR/nonce_seeker"
  printf '%s\nnvcc=%s\n' "$(gpu_build_key)" "$(nvcc_version "${NVCC:-nvcc}" 2>/dev/null || true)" | as_user tee "$GPU_INFO" >/dev/null
  ok "gpu_seeker + gpu_seeker.fatbin -> $BIN_DIR"
}

# environment the seeker runs with: the launcher exports the same
seeker_env() {
  gpu_params
  SEEKER_ENV=(GPU_SEEKER_FATBIN="$SEEKER_FATBIN" GPU_SEEKER_TARGET_MS="$GPU_TARGET_MS" GPU_SEEKER_BLOCKS_PER_SM="$GPU_BPS")
  if [[ -n "$GPU_DEVICES" ]]; then SEEKER_ENV+=(GPU_SEEKER_DEVICES="$GPU_DEVICES"); fi
  return 0
}
random_seed() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# gpu_selftest [--list|--check|--bench N] : run the installed seeker on a throw-away key (the key does not matter
# for the self-test); prints the seeker's output indented; returns its exit code
gpu_selftest() {
  local mode="${1:---check}"; shift || true
  [[ -x "$SEEKER_BIN" ]] || { warn "gpu_seeker is not installed ($SEEKER_BIN)"; return 3; }
  seeker_env
  local out rc=0
  out=$(as_user env "${SEEKER_ENV[@]}" NONCE_SEEKER_SEED="$(random_seed)" timeout 600 "$SEEKER_BIN" "$mode" "$@" 2>&1 </dev/null) || rc=$?
  while IFS= read -r l; do [[ -n "$l" ]] && printf '    %s\n' "$l"; done <<<"$out"
  return "$rc"
}

# =============================================================================
#  Settings: threads
# =============================================================================
SAVED_WORKERS=""
SAVED_HAD_GPU=0
gpu_summary() { # one line per GPU
  local i
  for i in "${!GPU_IDX[@]}"; do
    printf '    %sGPU %s%s  %s  %s(compute %s, %s MB)%s\n' "$C_W" "${GPU_IDX[$i]}" "$C_0" "${GPU_NAMES[$i]}" "$C_D" "${GPU_CCS[$i]:-?}" "${GPU_MEMS[$i]}" "$C_0"
  done
}
gpu_driver_check() { # warnings that save an hour of head-scratching
  local i cc need50=0 old=0
  for i in "${!GPU_CCS[@]}"; do
    cc="${GPU_CCS[$i]}"
    [[ -n "$cc" ]] || { warn "GPU ${GPU_IDX[$i]} (${GPU_NAMES[$i]}): compute capability unknown — the kernel is written for >= 7.5 (RTX 20xx and newer)"; continue; }
    if awk -v c="$cc" 'BEGIN{exit !(c >= 10)}'; then need50=1; fi
    if awk -v c="$cc" 'BEGIN{exit !(c < 7.5)}'; then old=1; fi
  done
  if (( old )); then warn "a GPU older than compute 7.5 (before RTX 20xx) is not supported by the kernel"; fi
  if (( need50 )) && [[ -n "$GPU_DRIVER" ]] && ! ver_ge "$GPU_DRIVER" 570; then
    warn "RTX 50xx (Blackwell) needs NVIDIA driver >= 570; this machine has $GPU_DRIVER — update the driver or the GPU seeker will not start"
  fi
  return 0
}

# choose_gpu: decides USE_GPU / GPU_DEVICES / GPU_INTENSITY (asks unless flags or saved settings decide)
choose_gpu() {
  step "GPU"
  detect_gpus
  if (( GPU_N == 0 )); then
    if [[ "$USE_GPU" == 1 ]]; then
      die "--gpu given but no NVIDIA GPU answers nvidia-smi${GPU_PCI_NAME:+ (lspci sees: $GPU_PCI_NAME — the driver is missing or not loaded)}. Install the NVIDIA driver (>= 570 for RTX 50xx), reboot, then retry."
    fi
    if [[ -n "$GPU_PCI_NAME" ]]; then
      warn "an NVIDIA device is present ($GPU_PCI_NAME) but the driver does not answer (nvidia-smi missing or failing)."
      warn "install the driver (Ubuntu: 'sudo ubuntu-drivers install', or nvidia-driver-570 or newer), reboot, and run '$0 install' again for the GPU."
    else
      log "no NVIDIA GPU found on this machine"
    fi
    USE_GPU=0; WITH_SEEKER=0
    ok "installing the CPU miner (same as proxima-miner.sh)"
    return 0
  fi
  ok "$GPU_N NVIDIA GPU(s), driver ${GPU_DRIVER:-?}:"
  gpu_summary
  gpu_driver_check
  if [[ -z "$USE_GPU" ]]; then
    if confirm "mine on the GPU(s)?" y; then USE_GPU=1; else USE_GPU=0; fi
  fi
  if [[ "$USE_GPU" != 1 ]]; then WITH_SEEKER=0; ok "GPU mining is off — CPU only"; return 0; fi
  WITH_SEEKER=1
  if (( GPU_N > 1 )) && [[ -z "$GPU_DEVICES" ]]; then
    ask GPU_DEVICES "GPUs to use: device numbers like 0,1 — or 'all'" "all"
  fi
  case "${GPU_DEVICES,,}" in all|"") GPU_DEVICES="";; esac
  if [[ -n "$GPU_DEVICES" ]]; then
    [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --gpu-devices '$GPU_DEVICES' (use 0,1 or all)"
    local d i have
    for d in ${GPU_DEVICES//,/ }; do
      have=0; for i in "${GPU_IDX[@]}"; do [[ "$i" == "$d" ]] && have=1; done
      (( have )) || die "there is no GPU $d on this machine (found: ${GPU_IDX[*]})"
    done
  fi
  if [[ -z "$GPU_INTENSITY" ]]; then
    if (( HAVE_TTY )); then
      printf '    low     short kernels: the desktop stays smooth, lower hashrate\n    normal  ~150 ms kernels (recommended for a dedicated rig / VPS)\n    high    ~300 ms kernels, more blocks in flight\n'
    fi
    ask GPU_INTENSITY "GPU intensity (low / normal / high)" "normal"
  fi
  case "$GPU_INTENSITY" in low|normal|high) ;; *) die "invalid --gpu-intensity '$GPU_INTENSITY' (low | normal | high)";; esac
  ok "GPU mining on — ${GPU_DEVICES:+GPU(s) $GPU_DEVICES}${GPU_DEVICES:-all GPUs}, intensity $GPU_INTENSITY, architectures: $(gpu_arch_list)"
}

choose_workers() {
  step "CPU threads"
  local n def; n=$(nproc 2>/dev/null || echo 1)
  if [[ "$USE_GPU" == 1 ]]; then
    # with a GPU the CPU threads are optional: 0 means GPU only (the GPU feeder needs hardly any CPU)
    def=0; if (( SAVED_HAD_GPU )) && [[ "$SAVED_WORKERS" =~ ^[0-9]+$ ]]; then def="$SAVED_WORKERS"; fi
    if [[ -z "$MINER_WORKERS" ]]; then ask MINER_WORKERS "extra CPU threads next to the GPU (this machine has $n cores; 0 = GPU only)" "$def"; fi
    [[ "$MINER_WORKERS" =~ ^[0-9]+$ ]] || die "invalid thread count: '$MINER_WORKERS'"
  else
    if [[ -z "$MINER_WORKERS" ]]; then ask MINER_WORKERS "CPU threads to mine with (this machine has $n)" "$([[ "$SAVED_WORKERS" =~ ^[1-9][0-9]*$ ]] && echo "$SAVED_WORKERS" || echo "$n")"; fi
    [[ "$MINER_WORKERS" =~ ^[0-9]+$ && "$MINER_WORKERS" -ge 1 ]] || die "invalid thread count: '$MINER_WORKERS' (at least 1 without a GPU)"
  fi
  (( MINER_WORKERS <= n )) || warn "more threads ($MINER_WORKERS) than cores ($n): extra threads only slow the others down"
  MINER_MAX_KHS="${MINER_MAX_KHS:-0}"
  [[ "$MINER_MAX_KHS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid --max-khs: '$MINER_MAX_KHS'"
  if [[ "$USE_GPU" == 1 ]]; then
    ok "GPU + $MINER_WORKERS CPU thread(s)$([[ "$MINER_MAX_KHS" != 0 ]] && echo ", CPU threads capped at $MINER_MAX_KHS KH/s")"
  else
    ok "$MINER_WORKERS thread(s)$([[ "$MINER_MAX_KHS" != 0 ]] && echo ", capped at $MINER_MAX_KHS KH/s")"
  fi
}

# =============================================================================
#  Launcher + service
# =============================================================================
# One small script is what systemd (or the background fallback) runs. It waits until your node
# is synced, then execs `proxi node mine`. The node URL is read from proxi.yaml at every start,
# so `set-node` only has to edit the profile.
write_launcher() {
  local streams=""
  [[ -n "$NO_PUBLIC" ]] || streams=$(IFS=,; echo "${PUBLIC_APIS[*]}")
  local body
  body=$(
    printf '#!/usr/bin/env bash\n# generated by proxima-gpu-miner.sh %s — regenerated on install/update/restart-after-change\n' "$SCRIPT_VERSION"
    printf 'WALLET_DIR=%q\nENV_FILE=%q\nPROXI=%q\nWORKERS=%q\nMAX_KHS=%q\nSTREAMS=%q\nUSE_GPU=%q\nSEEKER_PATH_DIR=%q\n' \
      "$WALLET_DIR" "$ENV_FILE" "$BIN_DIR/proxi" "$MINER_WORKERS" "${MINER_MAX_KHS:-0}" "$streams" "${USE_GPU:-0}" "$SEEKER_PATH_DIR"
    if [[ "${USE_GPU:-0}" == 1 ]]; then
      seeker_env
      # proxi spawns its seeker with a fixed command line: gpu_seeker is tuned through these variables
      local e; for e in "${SEEKER_ENV[@]}"; do printf 'export %q\n' "$e"; done
    fi
    cat <<'EOT'
cd "$WALLET_DIR" || exit 1
if [ -r "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
url=$(awk '/^[ \t]+(node_url|endpoint):/{print $2;exit}' proxi.yaml)
[ -n "$url" ] || { echo "no node_url in $WALLET_DIR/proxi.yaml"; exit 1; }

# wait (max ~1 h) until the node reports synced: mining on a node that is behind only wastes work
i=0
while [ "$i" -lt 360 ]; do
  if curl -fs -m 5 "$url/api/v1/sync_info" 2>/dev/null | grep -q '"synced": *true'; then echo "node $url is synced"; break; fi
  [ $((i % 6)) -eq 0 ] && echo "waiting for $url to be reachable and synced…"
  i=$((i + 1)); sleep 10
done
[ "$i" -lt 360 ] || echo "node $url not synced after 1 h — starting the miner anyway"

# -v: one "window ... at N H/s" line per round. Without it the miner prints a rate only when a round lasts >2 s,
# which at low difficulty is almost never (the dashboard then has no hashrate to show)
args=(node mine -v --workers "$WORKERS")
if [ "$USE_GPU" = 1 ]; then
  # proxi starts the program called `nonce_seeker` (found first in SEEKER_PATH_DIR: it is gpu_seeker),
  # hands it the job server address, token and key, restarts it if it dies and stops it with the miner
  export PATH="$SEEKER_PATH_DIR:$PATH"
  args+=(--seeker)
fi
[ "$MAX_KHS" != 0 ] && args+=(--max-hashrate-khs "$MAX_KHS")
[ -n "$STREAMS" ] && args+=(--stream "$STREAMS")
exec "$PROXI" "${args[@]}"
EOT
  )
  printf '%s\n' "$body" | as_user tee "$LAUNCHER" >/dev/null
  as_user chmod 700 "$LAUNCHER"
  ok "launcher: $LAUNCHER"
}

write_unit() {
  have_systemd || { warn "systemd not available (or --no-services): the miner will run as a background process"; return 0; }
  as_root tee "$UNIT_FILE" >/dev/null <<EOF
[Unit]
Description=Proxima CPU+GPU miner (proxi node mine -> $MINER_SVC)
After=network-online.target time-sync.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$WALLET_DIR
ExecStart=$LAUNCHER
Restart=always
RestartSec=15
TimeoutStopSec=30
KillSignal=SIGTERM
Nice=10

[Install]
WantedBy=multi-user.target
EOF
  as_root systemctl daemon-reload
  ok "$MINER_SVC.service"
}

unit_installed() { have_systemd && [[ -f "$UNIT_FILE" ]]; }

find_proc() { pgrep -n -f "^[^ ]*/proxi node mine" 2>/dev/null || true; }
svc_pid() {
  local p=""
  if unit_installed; then p=$(systemctl show -p MainPID --value "$MINER_SVC" 2>/dev/null || true)
  elif [[ -f "$PID_FILE" ]]; then p=$(cat "$PID_FILE" 2>/dev/null || true); fi
  if [[ ! "$p" =~ ^[0-9]+$ ]] || ! kill -0 "$p" 2>/dev/null; then p=$(find_proc); fi
  if [[ "$p" =~ ^[0-9]+$ && "$p" != 0 ]]; then echo "$p"; fi
  return 0
}
svc_state() { # active|inactive|running|stopped|none
  if unit_installed; then systemctl is-active "$MINER_SVC" 2>/dev/null || true; return 0; fi
  if [[ -n "$(svc_pid)" ]]; then echo running; return 0; fi
  if [[ -f "$LAUNCHER" ]]; then echo stopped; return 0; fi
  echo none
}

fallback_stop() {
  local pid i; pid=$(svc_pid)
  if [[ -n "$pid" ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    ok "stopped miner (pid $pid)"
  fi
  rm -f "$PID_FILE"
}
fallback_start() {
  if [[ -n "$(svc_pid)" ]]; then ok "miner already running (pid $(svc_pid))"; return 0; fi
  [[ -x "$LAUNCHER" ]] || die "no launcher — run: $0 install"
  # plain log file: keep it from growing forever
  if [[ -f "$LOG_FILE" ]] && (( $(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0) > 52428800 )); then
    as_user tail -c 2000000 "$LOG_FILE" | as_user tee "$LOG_FILE.new" >/dev/null; as_user mv -f "$LOG_FILE.new" "$LOG_FILE"
  fi
  spawn_detached "$WALLET_DIR" "$LOG_FILE" "$PID_FILE" "$LAUNCHER"
  ok "miner started in the background (pid $(cat "$PID_FILE" 2>/dev/null || echo ?), log $LOG_FILE)"
  warn "no systemd: the miner does not come back after a reboot — run '$0 start' again (or add an @reboot cron job)"
}

svc() { # svc start|stop|restart|enable
  local action="$1"
  if unit_installed; then
    case "$action" in
      enable)  as_root systemctl enable "$MINER_SVC" >/dev/null 2>&1; as_root systemctl restart "$MINER_SVC"; ok "enabled + started $MINER_SVC";;
      *)       as_root systemctl "$action" "$MINER_SVC"; ok "$action $MINER_SVC";;
    esac
  else
    case "$action" in
      stop)          fallback_stop;;
      restart)       fallback_stop; fallback_start;;
      start|enable)  fallback_start;;
    esac
  fi
}

# =============================================================================
#  Dashboard:  status  |  watch
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

dash_banner() {
  local title="  PROXIMA CPU+GPU MINER" stamp; stamp="$(date -u '+%F %T') UTC  "
  local pad=$((DASH_W - 4 - ${#title} - ${#stamp} - 2)); if (( pad < 1 )); then pad=1; fi
  printf '\n  %s╭%s╮%s\n' "$C_C" "$(rule $((DASH_W - 4)))" "$C_0"
  printf '  %s│%s %s◆%s%s%s%*s%s%s%s│%s\n' "$C_C" "$C_0" "$C_M$C_B" "$C_0" "$C_B$C_W" "$title" "$pad" "" "$C_D" "$stamp" "$C_C" "$C_0"
  printf '  %s╰%s╯%s\n' "$C_C" "$(rule $((DASH_W - 4)))" "$C_0"
}

# ---- network: what the public nodes say -------------------------------------
PUB_CUR=""; PUB_LRB=""
dash_network() {
  sect "NETWORK"
  if load_net_info; then
    kv "ledger" "${C_B}${C_W}${NET_DESC:-?}${C_0}"
    kv "genesis" "$(date -u -d "@$NET_GENESIS" '+%F %T' 2>/dev/null || echo "$NET_GENESIS") UTC   ${C_D}hash ${NET_HASH:0:12}…${C_0}"
    kv "nodes run" "${C_C}${NET_NODE_VER:-?}${C_0}   ${C_D}miner version ${NET_MINER_VER:-?}${C_0}"
    local ps u=""
    for u in "${LEDGER_APIS[@]}"; do ps=$(sync_info "$u"); [[ -n "$ps" ]] && break; done
    if [[ -n "${ps:-}" ]]; then
      PUB_CUR=$(jq -r '.current_slot // empty' <<<"$ps" 2>/dev/null || true)
      PUB_LRB=$(jq -r '.lrb_slot // empty' <<<"$ps" 2>/dev/null || true)
      kv "current slot" "${C_W}${PUB_CUR:-?}${C_0}   ${C_D}network LRB slot ${PUB_LRB:-?}${C_0}"
    fi
  else
    printf '  %s! no node answered for ledger info%s\n' "$C_Y" "$C_0"
  fi
}

# ---- your node: link quality, sync, ledger ----------------------------------
dash_node() {
  sect "YOUR NODE"
  kv "API" "${C_W}$NODE_URL${C_0}"
  if ! probe_node "$NODE_URL"; then
    kv "link" "$(dot_off) ${C_R}not reachable${C_0}   ${C_D}(node down, or its firewall blocks this VPS)${C_0}"
    note "the miner keeps retrying; check:  $0 check"
    return 0
  fi
  local rc=$C_G; if (( NP_RTT > 250 )); then rc=$C_Y; fi; if (( NP_RTT > 1000 )); then rc=$C_R; fi
  kv "link" "$(dot_on) ${C_G}reachable${C_0}   ${C_D}round trip${C_0} ${rc}${NP_RTT} ms${C_0}   ${C_D}${NP_VER:-?}${C_0}"
  if [[ "$NP_SYNCED" == true ]]; then kv "sync" "$(dot_on) ${C_G}synced${C_0}   ${C_D}slot ${NP_CUR:-?} · LRB ${NP_LRB:-?}${C_0}"
  else kv "sync" "$(dot_mid) ${C_Y}NOT synced${C_0}   ${C_D}slot ${NP_CUR:-?} · LRB ${NP_LRB:-?}${C_0}"; fi
  if [[ "$NP_LRB" =~ ^[0-9]+$ && "$PUB_LRB" =~ ^[0-9]+$ ]]; then
    local d=$((PUB_LRB - NP_LRB)); (( d < 0 )) && d=$((-d))
    if (( d <= 3 )); then kv "vs network" "${C_G}in step${C_0} ${C_D}(LRB differs by $d slot(s))${C_0}"
    else kv "vs network" "${C_Y}LRB differs by $d slots${C_0} ${C_D}(node behind, or on another ledger)${C_0}"; fi
  fi
  if [[ -n "$NET_GENESIS" && -n "$NP_GEN" && "$NET_GENESIS" != "$NP_GEN" ]]; then
    printf '  %s%s ✘ this node is on ANOTHER ledger (genesis %s) — it needs: proxima-setup.sh reset %s\n' "$C_B" "$C_R" "$NP_GEN" "$C_0"
  fi
}

# ---- miner: log + hashrate --------------------------------------------------
miner_log() {
  local o=""
  if unit_installed; then
    o=$(journalctl -u "$MINER_SVC" --no-pager -n 6000 -o cat 2>/dev/null || true)
    [[ -n "$o" ]] || o=$(sudo -n journalctl -u "$MINER_SVC" --no-pager -n 6000 -o cat 2>/dev/null || true)
  else
    o=$(tail -c 400000 "$LOG_FILE" 2>/dev/null || true)
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
  if ! grep -q -- 'node mine' <<<"$cmd"; then # launcher still waiting for the node to sync: show the configured values
    [[ "${MINER_WORKERS:-}" =~ ^[0-9]+$ ]] && workers="$MINER_WORKERS"
    cmd="node mine"; [[ "${USE_GPU:-0}" == 1 ]] && cmd="node mine --seeker"
    workers="$workers ${C_D}(waiting for the node to sync)${C_0}"
  fi
  seeker="no"; if grep -q -- '--seeker' <<<"$cmd"; then seeker="${C_G}yes (CUDA gpu_seeker)${C_0}"; fi
  cap=$(grep -oE -- '--max-hashrate-khs +[0-9.]+' <<<"$cmd" | awk '{print $2" KH/s cap"}' || true)
  kv "CPU threads" "${C_W}$workers${C_0}   ${C_D}GPU seeker:${C_0} $seeker${cap:+   ${C_D}$cap${C_0}}"

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
    kv "hashrate" "${C_B}${C_G}$(fmt_rate "$rate")${C_0} ${C_D}$([[ "${USE_GPU:-0}" == 1 ]] && echo "CPU+GPU total")${C_0}   ${C_D}avg${C_0} ${C_W}$(fmt_rate "${avg:-0}")${C_0}  ${C_D}peak${C_0} ${C_W}$(fmt_rate "${peak:-0}")${C_0}"
    local sp; sp=$(tr ' ' '\n' <<<"$samples" | awk 'NF' | spark)
    [[ -n "$sp" ]] && kv "recent" "${C_C}$sp${C_0}"
  else
    kv "hashrate" "${C_Y}no samples yet${C_0} ${C_D}(first mining round pending, or the launcher is older than 1.0.1: run '$0 update')${C_0}"
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

# ---- GPU: live card state from nvidia-smi + what the seeker says in the miner log
dash_gpu() {
  sect "GPU"
  if [[ "${USE_GPU:-0}" != 1 ]]; then
    kv "mode" "${C_D}GPU mining is off — CPU only${C_0}   ${C_D}(turn on: $0 gpu on)${C_0}"
    return 0
  fi
  local smi="" line idx name util temp pw plim mu mt clk
  if command -v nvidia-smi >/dev/null 2>&1; then
    smi=$(timeout 8 nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total,clocks.sm \
          --format=csv,noheader,nounits 2>/dev/null || true)
  fi
  if [[ -z "$smi" ]]; then
    kv "cards" "${C_R}nvidia-smi does not answer${C_0} ${C_D}(driver not loaded? reboot after a driver update)${C_0}"
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      IFS=',' read -r idx name util temp pw plim mu mt clk <<<"$line"
      idx=$(trim "$idx"); name=$(trim "$name"); util=$(trim "$util"); temp=$(trim "$temp"); pw=$(trim "$pw"); plim=$(trim "$plim")
      mu=$(trim "$mu"); mt=$(trim "$mt"); clk=$(trim "$clk")
      [[ "$util" =~ ^[0-9]+$ ]] || util=0
      local tc=$C_G; [[ "$temp" =~ ^[0-9]+$ ]] && { (( temp >= 75 )) && tc=$C_Y; (( temp >= 85 )) && tc=$C_R; }
      [[ -z "$GPU_DEVICES" ]] || [[ ",$GPU_DEVICES," == *",$idx,"* ]] || continue
      kv "GPU $idx" "${C_W}${name}${C_0}   ${C_D}${mu:-?}/${mt:-?} MiB${C_0}"
      printf '  %s%-17s%s %s %s%3s%%%s  %s%s°C%s  %s%s W%s%s  %s MHz%s\n' "$C_D" "" "$C_0" "$(bar "$util" 100 22 "$C_G")" "$C_W" "$util" "$C_0" \
        "$tc" "${temp:-?}" "$C_0" "$C_W" "${pw%.*}" "$C_0" "${plim:+${C_D}/ ${plim%.*}}" "${clk:-?}" "$C_0"
    done <<<"$smi"
  fi
  local log; log=$(miner_log)
  [[ -n "$log" ]] || return 0
  local rate="" samples="" ok_gpus="" bad="" sv=0 rj=0 refuse="" unreach="" restarts=0 fatal="" key val
  while IFS=$'\t' read -r key val; do
    case "$key" in rate) rate=$val;; samples) samples=$val;; ok) ok_gpus=$val;; bad) bad=$val;; sv) sv=$val;; rj) rj=$val;;
                   refuse) refuse=$val;; unreach) unreach=$val;; restarts) restarts=$val;; fatal) fatal=$val;; esac
  done < <(awk '
    index($0,"[seeker]") {
      if (match($0,/: [0-9]+ H\/s/)) { r=substr($0,RSTART+2,RLENGTH-2); sub(/ H\/s/,"",r); sm[++ns]=r }
      if (match($0,/gpu[0-9]+ .*: kernel agrees/)) { g=substr($0,RSTART,RLENGTH); sub(/: kernel agrees/,"",g); okg[g]=1 }
      if (index($0,"NOT USED")) { b=$0; sub(/.*\[seeker\] [0-9:]+ /,"",b) }
      if (index($0,"SOLVED with nonce")) sv++
      if (index($0,"REJECTED nonce")) rj++
      if (index($0,"REFUSING job")) { f=$0; sub(/.*\[seeker\] [0-9:]+ /,"",f) }
      if (index($0,"cannot reach the miner")) un=1; else if (match($0,/: [0-9]+ H\/s/)) un=0
      if (index($0,"no usable NVIDIA GPU")||index($0,"cannot load the key")) fa=$0
    }
    index($0,"nonce seeker exited") { rs++ }
    END { if (ns>0) { print "rate\t" sm[ns]; s=""; st=(ns>40)?ns-39:1; for(i=st;i<=ns;i++) s=s sm[i] " "; print "samples\t" s }
          o=""; for (k in okg) o=o (o==""?"":", ") k; print "ok\t" o; print "bad\t" b; print "sv\t" sv+0; print "rj\t" rj+0
          print "refuse\t" f; print "unreach\t" un+0; print "restarts\t" rs+0; print "fatal\t" fa }' <<<"$log") || true

  if [[ -n "$ok_gpus" ]]; then kv "kernel check" "${C_G}passed${C_0} ${C_D}against the CPU reference on: ${ok_gpus}${C_0}"
  elif [[ -n "$bad" ]]; then kv "kernel check" "${C_R}${bad:0:80}${C_0}"
  else kv "kernel check" "${C_Y}not seen in the visible log yet${C_0}"; fi
  if [[ "$rate" =~ ^[0-9]+$ && "$rate" -gt 0 ]]; then
    local avg peak; read -r avg peak < <(awk '{n=0;s=0;m=0; for(i=1;i<=NF;i++){ s+=$i;n++; if($i>m)m=$i } printf "%d %d", (n?s/n:0), m}' <<<"$samples") || true
    kv "GPU hashrate" "${C_B}${C_G}$(fmt_rate "$rate")${C_0}   ${C_D}avg${C_0} ${C_W}$(fmt_rate "${avg:-0}")${C_0}  ${C_D}peak${C_0} ${C_W}$(fmt_rate "${peak:-0}")${C_0}"
    local sp; sp=$(tr ' ' '\n' <<<"$samples" | awk 'NF' | spark)
    [[ -n "$sp" ]] && kv "recent" "${C_C}$sp${C_0}"
  else
    kv "GPU hashrate" "${C_Y}no rate yet${C_0} ${C_D}(the seeker reports every 10 s once the node is synced and a mining round runs)${C_0}"
  fi
  kv "GPU finds" "${C_G}${sv}${C_0} ${C_D}solved ·${C_0} ${C_R}${rj}${C_0} ${C_D}rejected by the miner (in visible log)${C_0}"
  if (( restarts > 0 )); then kv "restarts" "${C_Y}${restarts}${C_0} ${C_D}seeker restart(s) in the visible log${C_0}"; fi
  if [[ -n "$refuse" ]]; then kv "last problem" "${C_Y}${refuse:0:90}${C_0}"; fi
  if [[ "$unreach" == 1 ]]; then kv "last problem" "${C_Y}the seeker cannot reach the miner's job server${C_0}"; fi
  if [[ -n "$fatal" ]]; then kv "last problem" "${C_R}${fatal:0:90}${C_0}"; fi
  return 0
}

dash_software() {
  sect "SOFTWARE"
  local pv; pv=$(installed_version)
  local mode="CPU only"; [[ "${USE_GPU:-0}" != 1 ]] || mode="CPU + GPU"
  kv "installed" "proxi ${C_C}${pv:-?}${C_0}   ${C_D}script $SCRIPT_VERSION · $mode${C_0}"
  if [[ "${USE_GPU:-0}" == 1 && -f "$GPU_INFO" ]]; then
    local ga gn; ga=$(sed -nE '1s/.*archs=(.*)$/\1/p' "$GPU_INFO"); gn=$(sed -n 's/^nvcc=//p' "$GPU_INFO" | head -n1)
    kv "gpu seeker" "${C_C}gpu_seeker${C_0}   ${C_D}CUDA kernel for sm_${ga// /, sm_}${gn:+ · built with nvcc $gn}${C_0}"
    if gpu_stale 2>/dev/null; then kv "" "${C_Y}the seeker is older than this script or the proxima source — run: $0 update${C_0}"; fi
  fi
  if [[ -d "$SRC_DIR/.git" ]]; then
    git -C "$SRC_DIR" fetch --quiet --tags origin 2>/dev/null || true
    local rv oh bh ur="origin/$BRANCH"
    git -C "$SRC_DIR" rev-parse --verify --quiet "refs/remotes/$ur" >/dev/null 2>&1 || ur="$BRANCH"
    rv=$(git -C "$SRC_DIR" show "$ur:global/version.go" 2>/dev/null | grep -oE 'Version\s+=\s+"[^"]+"' | cut -d'"' -f2 || true)
    oh=$(git -C "$SRC_DIR" rev-parse "$ur^{commit}" 2>/dev/null || true)
    bh=$(cat "$STAMP_FILE" 2>/dev/null || true)
    if [[ -n "$oh" && -n "$bh" && "$oh" != "$bh" ]]; then
      kv "upstream" "${C_Y}$BRANCH has ${rv:-?} (${oh:0:8}) — update available: ${C_B}$0 update${C_0}"
    else kv "upstream" "${C_G}up to date${C_0} ${C_D}($BRANCH ${rv:-?})${C_0}"; fi
  fi
}

dash_footer() {
  local me="${0##*/}" d="${C_D}·${C_0}"
  printf '\n  %s' "$C_D"; rule $((DASH_W - 4)); printf '%s\n' "$C_0"
  printf '  %s%s logs%s %s %s%s watch [secs]%s %s %s%s check%s %s %s%s gpu%s %s %s%s bench%s %s %s%s set-node URL%s %s %s%s key%s %s %s%s update%s\n' \
    "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0"
}

cmd_status() {
  load_install
  PUB_CUR=""; PUB_LRB=""
  dash_banner
  dash_network
  dash_node
  dash_wallet
  dash_miner
  dash_gpu
  dash_software
  dash_footer
}

# live dashboard, refreshed in place
cmd_watch() {
  local n="${1:-10}"; [[ "$n" =~ ^[0-9]+$ && "$n" -ge 3 ]] || n=10
  [[ -t 1 ]] || die "watch needs a terminal"
  load_install
  tput civis 2>/dev/null || true
  trap 'tput cnorm 2>/dev/null || true; printf "\n"; exit 0' INT TERM
  local out
  while true; do
    out=$( (cmd_status) 2>&1 | sed 's/$/\x1b[K/' || true)
    printf '\e[H%s\n\n  %srefreshing every %ss — Ctrl-C to quit%s\e[K\n\e[J' "$out" "$C_D" "$n" "$C_0"
    sleep "$n"
  done
}

cmd_logs() {
  load_install
  # not 'exec': as_root is a shell function. Ctrl-C ends the follow; that is not an error.
  if unit_installed; then as_root journalctl -u "$MINER_SVC" -f -n 100 -o cat || true
  else tail -n 100 -F "$LOG_FILE" | tr '\r' '\n' || true; fi
}

# =============================================================================
#  check: can this VPS use your node?  (works before anything is installed)
# =============================================================================
cmd_check() {
  local url="${NODE_URL:-}"
  if [[ -z "$url" && -f "$WALLET_DIR/proxi.yaml" ]]; then url=$(profile_node_url); fi
  url=$(norm_url "${url:-$DEFAULT_NODE_URL}")
  command -v curl >/dev/null || die "curl is required"
  command -v jq   >/dev/null || die "jq is required (apt-get install -y jq)"
  NODE_URL="$url"; load_conf; setup_ledger_apis
  step "Connectivity: this VPS -> $url"
  local host port; host="${url#*//}"; host="${host%%/*}"; port="${host##*:}"; host="${host%%:*}"
  [[ "$port" =~ ^[0-9]+$ ]] || port=$([[ "$url" == https* ]] && echo 443 || echo 80)
  if timeout 6 bash -c ":</dev/tcp/$host/$port" 2>/dev/null; then ok "TCP $host:$port is open"
  else warn "TCP $host:$port does not answer — the node's firewall must allow this VPS ($(public_ip || true)) on port $port"; fi
  if probe_node "$url"; then
    ok "API answers in ${NP_RTT} ms — ${NP_VER:-?}, id ${NP_ID:0:16}…"
    if [[ "$NP_SYNCED" == true ]]; then ok "node is synced (slot ${NP_CUR:-?}, LRB ${NP_LRB:-?})"; else warn "node is NOT synced (slot ${NP_CUR:-?}, LRB ${NP_LRB:-?})"; fi
  else warn "no valid answer from $url/api/v1/sync_info"; fi
  local pg; pg=$(public_genesis)
  if [[ -n "$pg" ]]; then
    if [[ -n "$NP_GEN" && "$NP_GEN" == "$pg" ]]; then ok "same ledger as the public network (genesis $pg)"
    elif [[ -n "$NP_GEN" ]]; then warn "DIFFERENT ledger: node genesis $NP_GEN, public network $pg"; fi
  fi
  if [[ -z "$NO_PUBLIC" ]]; then
    local i n=0
    for i in "${!PUBLIC_APIS[@]}"; do
      if curl -fs -m 5 "${PUBLIC_APIS[$i]}/api/v1/sync_info" >/dev/null 2>&1; then n=$((n + 1)); ok "public ${PUBLIC_NAMES[$i]} (${PUBLIC_APIS[$i]}) reachable"
      else warn "public ${PUBLIC_NAMES[$i]} (${PUBLIC_APIS[$i]}) NOT reachable"; fi
    done
    (( n > 0 )) || warn "no public node reachable: the wallet's ledger-library witness check needs at least one (or use --no-public)"
  fi
  if command -v timedatectl >/dev/null; then
    if timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes; then ok "system clock is NTP-synchronized"
    else warn "system clock is NOT NTP-synchronized — ledger time is real time; 'install' will try to enable NTP"; fi
  fi
  local cores mem; cores=$(nproc 2>/dev/null || echo ?); mem=$(awk '/^MemTotal:/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo ?)
  ok "this VPS: $cores core(s), ${mem} MB RAM, public IP $(public_ip || echo '?')"
  step "GPU"
  if gpu_report; then
    if [[ -x "$SEEKER_BIN" && -f "$SEEKER_FATBIN" ]]; then
      log "running the kernel self-test on every GPU …"
      local rc=0; gpu_selftest --check || rc=$?
      if (( rc == 0 )); then ok "GPU seeker installed, kernel agrees with the CPU reference"; else warn "GPU self-test failed (exit $rc)"; fi
    else log "the GPU seeker is not installed yet — 'install' builds it"; fi
  fi
}

# =============================================================================
#  GPU commands:  gpu [on|off|test]   |   bench
# =============================================================================
libcuda_present() { ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1' || compgen -G '/usr/lib*/*/libcuda.so.1' >/dev/null || compgen -G '/usr/lib*/libcuda.so.1' >/dev/null; }

gpu_report() { # what a person needs to know about the GPU side of this machine
  detect_gpus
  if (( GPU_N == 0 )); then
    if [[ -n "$GPU_PCI_NAME" ]]; then warn "NVIDIA device present ($GPU_PCI_NAME) but nvidia-smi does not answer: the driver is missing or not loaded"
    else warn "no NVIDIA GPU found (nvidia-smi missing or no card)"; fi
    return 1
  fi
  ok "$GPU_N NVIDIA GPU(s), driver ${GPU_DRIVER:-?}"
  gpu_summary
  gpu_driver_check
  if libcuda_present; then ok "libcuda.so.1 found (the CUDA driver library the seeker loads)"
  else warn "libcuda.so.1 not found — install the NVIDIA driver package that provides it (nvidia-driver-570 or newer; not just nvidia-utils)"; fi
  local need nv; need=$(need_cuda_ver)
  if nv=$(find_nvcc "$need"); then ok "CUDA compiler: nvcc $(nvcc_version "$nv") ($nv)"
  else log "no CUDA compiler >= $need yet — 'install' downloads a private copy (~85 MB)"; fi
  if g++ --version >/dev/null 2>&1; then ok "host compiler g++ $(g++ -dumpversion 2>/dev/null)"; else warn "g++ is missing (needed by nvcc; 'install' adds it)"; fi
  return 0
}

cmd_gpu() {
  local sub="${ARGS[0]:-}"
  case "$sub" in
    test)
      load_install
      step "GPU kernel self-test (every GPU against the CPU reference)"
      gpu_report || true
      local rc=0; gpu_selftest --check || rc=$?
      if (( rc == 0 )); then ok "all GPUs agree with the CPU reference"; else die "self-test failed (exit $rc) — the lines above say why"; fi;;
    on)
      load_install
      [[ -x "$SEEKER_BIN" && -f "$SEEKER_FATBIN" ]] || die "the GPU seeker is not installed — run: $0 install --gpu"
      detect_gpus; (( GPU_N > 0 )) || die "no NVIDIA GPU answers nvidia-smi"
      USE_GPU=1; WITH_SEEKER=1
      local rc=0; gpu_selftest --check || rc=$?; (( rc == 0 )) || die "GPU self-test failed (exit $rc) — not switching the GPU on"
      save_conf; write_launcher; write_unit; svc restart; ok "GPU mining is on";;
    off)
      load_install
      USE_GPU=0
      if [[ ! "$MINER_WORKERS" =~ ^[0-9]+$ || "$MINER_WORKERS" -lt 1 ]]; then MINER_WORKERS=$(nproc 2>/dev/null || echo 1); fi
      save_conf; write_launcher; write_unit; svc restart; ok "GPU mining is off — $MINER_WORKERS CPU thread(s)";;
    ""|show|status|set)
      load_install
      step "GPU setup"
      gpu_report || true
      kv "mining on GPU" "$([[ "${USE_GPU:-0}" == 1 ]] && echo "${C_G}yes${C_0}" || echo "${C_Y}no${C_0}")"
      kv "devices" "${GPU_DEVICES:-all}"
      kv "intensity" "${GPU_INTENSITY:-normal}"
      kv "CPU threads" "$MINER_WORKERS"
      if [[ -x "$SEEKER_BIN" ]]; then
        kv "seeker" "$SEEKER_BIN"; [[ -f "$GPU_INFO" ]] && kv "built from" "$(head -n1 "$GPU_INFO") $(sed -n 2p "$GPU_INFO")"
        gpu_selftest --list || true
      else kv "seeker" "${C_D}not installed${C_0}"; fi;;
    *) die "usage: $0 gpu [on|off|test]   (change devices / intensity with --gpu-devices L --gpu-intensity low|normal|high)";;
  esac
}

cmd_gpu_tune() { # `gpu --gpu-devices 0 --gpu-intensity high`: change and restart
  load_install
  [[ -x "$SEEKER_BIN" ]] || die "the GPU seeker is not installed — run: $0 install --gpu"
  case "${GPU_INTENSITY:-normal}" in low|normal|high) ;; *) die "invalid --gpu-intensity '$GPU_INTENSITY' (low | normal | high)";; esac
  case "${GPU_DEVICES,,}" in all) GPU_DEVICES="";; esac
  [[ -z "$GPU_DEVICES" || "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --gpu-devices '$GPU_DEVICES' (use 0,1 or all)"
  USE_GPU=1; WITH_SEEKER=1
  save_conf; write_launcher; write_unit; svc restart
  ok "GPU settings applied: devices ${GPU_DEVICES:-all}, intensity ${GPU_INTENSITY:-normal}"
}

cmd_bench() {
  local secs="${ARGS[0]:-10}"; [[ "$secs" =~ ^[0-9]+$ && "$secs" -ge 3 ]] || secs=10
  load_install
  [[ -x "$SEEKER_BIN" ]] || die "the GPU seeker is not installed — run: $0 install --gpu"
  step "GPU benchmark ($secs s per GPU)"
  local was; was=$(svc_state)
  case "$was" in active|running) log "pausing the miner for the measurement"; svc stop 2>/dev/null || true;; *) was="";; esac
  local rc=0
  gpu_selftest --bench "$secs" || rc=$?
  if [[ -n "$was" ]]; then svc start || true; fi
  (( rc == 0 )) || die "benchmark failed (exit $rc)"
  log "these figures are what the kernel does on this card with intensity '${GPU_INTENSITY:-normal}'; real mining adds job-switch overhead"
}

# =============================================================================
#  Commands
# =============================================================================
summary() {
  local holder; holder=$(profile_holder)
  step "Done — $([[ "${USE_GPU:-0}" == 1 ]] && echo "CPU+GPU" || echo "CPU") miner is set up"
  cat <<EOT
  ${C_B}What runs${C_0}
    node (remote) : $NODE_URL
    miner         : $(if [[ "${USE_GPU:-0}" == 1 ]]; then echo "GPU ($([[ -n "$GPU_DEVICES" ]] && echo "device(s) $GPU_DEVICES" || echo "all $GPU_N"), intensity $GPU_INTENSITY) + "; fi)$MINER_WORKERS CPU thread(s)$([[ "${MINER_MAX_KHS:-0}" != 0 ]] && echo ", cap $MINER_MAX_KHS KH/s") — $(if unit_installed; then echo "systemd: $MINER_SVC"; else echo "background process (pid file $PID_FILE)"; fi)
    wallet        : $WALLET_DIR   holder ${C_W}$holder${C_0}

  ${C_B}Everyday commands${C_0}   (shortcut: ${C_C}${SELF_SHORT:-$0}${C_0} <command>)
    status | watch   dashboard     logs   follow the miner     check   test node link + GPU
    gpu [on|off|test]   GPU setup / self-test     bench   measure the GPU hashrate
    set-node URL     use another node     key   change wallet key     update   rebuild from master

  ${C_B}Rewards${C_0}
EOT
  if [[ "$KEY_MODE" == generate ]]; then
    cat <<EOT
    This VPS mines with a NEW key, so rewards collect in its own wallet. ${C_Y}Back up $WALLET_DIR/proxima.key.${C_0}
    Sweep to your main wallet:  cd $WALLET_DIR && $BIN_DIR/proxi node send_to_wallet <amount> <main holder ID>
EOT
  else
    printf '    Rewards go to the wallet of the key you imported (holder %s…).\n' "${holder:0:16}"
  fi
  printf '\n'
}

SELF_SHORT=""
pick_self_short() {
  if [[ -x /usr/local/bin/proxima-gpu-miner ]]; then SELF_SHORT=proxima-gpu-miner
  elif [[ -x "$MINER_HOME/proxima-gpu-miner.sh" ]]; then SELF_SHORT="$MINER_HOME/proxima-gpu-miner.sh"; fi
}

# the CPU-only edition (proxima-miner.sh) may be running from the same --home: this miner replaces it
retire_cpu_service() {
  if have_systemd && [[ -f "$OLD_CPU_UNIT" ]]; then
    log "replacing the CPU-only miner service $OLD_CPU_SVC (same wallet, same key)"
    as_root systemctl disable --now "$OLD_CPU_SVC" 2>/dev/null || true
    as_root rm -f "$OLD_CPU_UNIT"; as_root systemctl daemon-reload
  fi
  return 0
}

# GPU part of install/update: toolchains, build, install, self-test. Returns 1 when the GPU cannot be used.
gpu_self_check_or_fallback() {
  step "GPU self-test: kernel vs. CPU reference"
  local rc=0; gpu_selftest --check || rc=$?
  if (( rc == 0 )); then ok "every GPU reproduces the CPU reference output"; return 0; fi
  warn "the GPU self-test failed (exit $rc) — see the lines above. Possible causes: driver too old for this card (RTX 50xx: >= 570), no libcuda, GPU busy/hung."
  warn "falling back to CPU-only mining; fix the cause, then run: $0 gpu on"
  USE_GPU=0; WITH_SEEKER=0
  if [[ ! "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then MINER_WORKERS=$(nproc 2>/dev/null || echo 1); fi
  return 1
}

cmd_install() {
  step "Proxima CPU+GPU miner $SCRIPT_VERSION — user=$RUN_USER home=$MINER_HOME branch=$BRANCH"
  local forced_gpu="$USE_GPU"
  load_conf
  # a saved "GPU off" is re-asked (the driver may have been installed since); a saved "GPU on" is kept
  SAVED_HAD_GPU=0
  if [[ -z "$forced_gpu" ]]; then if [[ "$USE_GPU" == 1 ]]; then SAVED_HAD_GPU=1; else USE_GPU=""; fi; fi
  # does this machine have a GPU?  decides which build tools preflight / ensure_deps need
  detect_gpus
  if (( GPU_N > 0 )) && [[ "$USE_GPU" != 0 ]]; then WITH_SEEKER=1; fi
  preflight
  ensure_deps
  # remembered answers become the prompt defaults of a re-run
  if [[ -f "$WALLET_DIR/proxi.yaml" ]]; then SAVED_NODE_URL=$(profile_node_url); fi
  SAVED_WORKERS="$MINER_WORKERS"
  choose_node
  setup_ledger_apis
  choose_key
  choose_gpu
  if [[ "$USE_GPU" != 1 && ! "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then MINER_WORKERS=""; fi
  choose_workers
  check_clock
  fetch_source; check_compat
  ensure_go
  build_proxi
  if [[ "$USE_GPU" == 1 ]]; then ensure_cuda; ensure_rust; build_gpu_seeker; fi
  retire_cpu_service
  svc stop 2>/dev/null || true
  install_binary
  if [[ "$USE_GPU" == 1 ]]; then
    install_gpu_files
    if ! gpu_self_check_or_fallback; then ok "continuing as a CPU miner with $MINER_WORKERS thread(s)"; fi
  fi
  apply_key
  save_conf
  write_launcher
  write_unit
  svc enable
  pick_self_short
  summary
  sleep 4
  if [[ -t 1 ]]; then cmd_status || true; fi
}

cmd_set_node() {
  load_install
  local url="${ARGS[0]:-$NODE_URL_ARG}"
  [[ -n "$url" ]] || die "usage: $0 set-node http://IP:8000"
  NODE_URL="$url"; choose_node
  profile_apply "$NODE_URL"
  setup_ledger_apis
  svc restart
  ok "miner now uses $NODE_URL"
}

cmd_key() {
  load_install
  [[ -x "$BIN_DIR/proxi" ]] || die "proxi not found in $BIN_DIR — run: $0 install"
  choose_key
  [[ "$KEY_MODE" == keep ]] && { ok "key unchanged"; return 0; }
  svc stop 2>/dev/null || true
  apply_key
  write_launcher
  svc start
}

launcher_stale() { [[ -f "$LAUNCHER" ]] && ! grep -q "generated by proxima-gpu-miner.sh $SCRIPT_VERSION " "$LAUNCHER"; }

cmd_update() {
  step "Updating — branch $BRANCH"
  load_install
  [[ "$USE_GPU" == 1 ]] && WITH_SEEKER=1
  preflight; ensure_deps
  local old_v; old_v=$(installed_version)
  if [[ "$USE_GPU" == 1 ]]; then
    [[ "$MINER_WORKERS" =~ ^[0-9]+$ ]] || MINER_WORKERS=0
  else
    [[ "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]] || MINER_WORKERS=$(nproc 2>/dev/null || echo 1)
  fi
  fetch_source
  check_compat
  local have need_proxi=0 need_gpu=0
  have=$(cat "$STAMP_FILE" 2>/dev/null || true)
  if [[ -z "$have" || "$have" != "$(as_user git -C "$SRC_DIR" rev-parse HEAD)" || -n "${FORCE_REBUILD:-}" ]]; then need_proxi=1; fi
  if [[ "$USE_GPU" == 1 ]] && { gpu_stale || [[ -n "${FORCE_REBUILD:-}" ]]; }; then need_gpu=1; fi
  if (( ! need_proxi && ! need_gpu )) && launcher_stale; then
    log "the launcher is from an older script version: refreshing it (no rebuild needed)"
    retire_cpu_service; install_self; save_conf; write_launcher; write_unit; svc restart; ok "launcher + script updated, miner restarted"; return 0
  fi
  if (( ! need_proxi && ! need_gpu )); then
    ok "already running $old_v ($(printf '%.10s' "$have"))$([[ "$USE_GPU" == 1 ]] && echo ", GPU seeker up to date") — nothing to do"; return 0
  fi
  if (( need_proxi )); then ensure_go; build_proxi; log "upgrade: ${old_v:-none} -> $SRC_VERSION"; fi
  if (( need_gpu )); then ensure_cuda; ensure_rust; build_gpu_seeker; fi
  retire_cpu_service
  svc stop 2>/dev/null || true
  if (( need_proxi )); then install_binary; fi
  if (( need_gpu )); then install_gpu_files; gpu_self_check_or_fallback || true; fi
  migrate_wallet_profile
  save_conf; write_launcher; write_unit
  svc start
  ok "updated to $SRC_VERSION"
}

cmd_uninstall() {
  load_install 2>/dev/null || load_conf
  confirm "Stop and remove the miner service and the proxi / gpu_seeker binaries? (the wallet in $WALLET_DIR is kept)" y || exit 0
  svc stop 2>/dev/null || true
  if have_systemd && [[ -f "$UNIT_FILE" ]]; then
    as_root systemctl disable "$MINER_SVC" 2>/dev/null || true
    as_root rm -f "$UNIT_FILE"; as_root systemctl daemon-reload
  fi
  as_root rm -f "$BIN_DIR/proxi" "$BIN_DIR/proxi.prev" "$SEEKER_BIN" "$SEEKER_FATBIN"
  if [[ -L /usr/local/bin/proxima-gpu-miner ]]; then as_root rm -f /usr/local/bin/proxima-gpu-miner; fi
  rm -rf "$SEEKER_PATH_DIR" "$GPU_INFO" "$GPU_SRC" "$GPU_TARGET" "$GPU_BUILD"
  rm -f "$LAUNCHER" "$PID_FILE"
  ok "removed. Your key and data remain in $MINER_HOME"
  log "the private toolchains stay in $MINER_HOME/.toolchain (Go, Rust, CUDA compiler: ~1.5 GB) — they are reused by a later install"
  log "to delete everything (the Go module cache is read-only):  chmod -R u+w $MINER_HOME && rm -rf $MINER_HOME   ${C_Y}— back up proxima.key first${C_0}"
}

usage() {
  if [[ -n "$SELF_PATH" ]]; then awk 'NR>1 && /^#/{print; next} NR>1{exit}' "$SELF_PATH" | sed 's/^# \{0,1\}//' | grep -v '^====='
  else # piped (curl | bash): the file is not there to re-read
    cat <<EOT
proxima-gpu-miner.sh $SCRIPT_VERSION — standalone CPU + NVIDIA GPU (RTX 40xx / 50xx) Proxima miner, mines against a remote node

  install | status | watch [secs] | check | gpu [on|off|test] | bench [secs] | logs | start | stop | restart
  set-node URL | key | update | uninstall | help

  options: --node-url URL  --key-file FILE  --generate-key  --workers N (0 = GPU only)  --max-khs X  --no-public
           --gpu | --no-gpu  --gpu-devices 0,1  --gpu-intensity low|normal|high  --gpu-arch "89 120"
           --home DIR  --branch NAME  --bin-dir DIR  --no-services  --no-color  -y
  Save the script to a file (or run 'install' once) to see the full manual: proxima-gpu-miner.sh help
EOT
  fi
}

# >>> gpu-seeker payload (generated by gpu-seeker/tools/embed.py - do not edit by hand)
gpu_payload() {
  cat <<'GPU_SEEKER_PAYLOAD_EOF'
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
Do/DyDG2FNlbxEZkRAf27aObxWVnVuIW8sz2CKOR8DEVrjrLH1KvUtwoFqlDGSGnZ+tNJSIwwb3J
qOf0t04bEFmmOpv6Jr2rb0KmLDn0plSKrFvPix8x9VMkGHO8SrQ85qJLqXQ0K+980D94Glj/Jo/Q
GQJrwtd/mxye9PvJvyQHR/qbE/x80pfP+CdjZj2jeLiq2T3LqTwIFn3gLJYCY8z9tAlOogzz5UA+
wvmRJE5XzSDtaVFav5NP/uDerrCs80kwlpViu32iILsovQb/fY1NhlZP5NSUghLLv6OV9LsxhjXB
3ZKPh9koNKXiu353bmfZQRM64GivDhOF5Ckbe9Qrj472iaE13zUqyUNOUI1C+qx3jqrpG6pEqH6T
EXoZ5gzrXTO9QciXSdqmGyMG5aU3c0GW7tJO0o2Fg3PiFK4Q5/z6b+GK1y9TTBMAPUhZyKJ3UV1s
m2bb0nx3Uye95uxEJnxx3Hat6Co8/lSnJbRwtUrgT9ikfJa027j/ka4osRh616Js3ORxHLVL4LmI
5PuyOQ9PvXDBuWu9Xor/H3EcJs7ApZv/iiyI8187zwXdXUO++fhGkayLJoDzuaQsD01pXdgpmlE4
JJhkmDGLNhP44Sm8b3l9bd2KWekwjmX/uuH2oNljuvEysh22pcavQmdrZxPuH0LzD/kMoQaGZEuY
z4qqpswgaq0eTopb7eyJsd8wnZCGcHF66TqL1+Qfcq7hG+8Baxb8aHLfz5whVm8WVLB/mbJXKPw+
bABvDLzXLo6zN33uAT8kEwh7zJCn5tOxM1/WaGLeLiK4LlJY/xLGcto5e8aT/T49uY9P9p/15Cm9
7Y0e567O6O7aXbtrd+2u3bW7dtfu2l27a3ftrt21u3bX7tpdu2t37a7dtbt21+7aXbtrd+2u3bW7
dtfu2l27a3ftrv+T6x99XEk7ALgBAA==
GPU_SEEKER_PAYLOAD_EOF
}
# <<< gpu-seeker payload

NODE_URL_ARG="${NODE_URL:-}"
case "$CMD" in
  install)         cmd_install;;
  status)          cmd_status;;
  watch|dashboard) cmd_watch "${ARGS[0]:-10}";;
  check)           cmd_check;;
  gpu)             if [[ -z "${ARGS[0]:-}" && "$GPU_TUNE" == 1 ]]; then cmd_gpu_tune; else cmd_gpu; fi;;
  bench)           cmd_bench;;
  logs)            cmd_logs;;
  start|stop|restart) load_install; svc "$CMD";;
  set-node)        cmd_set_node;;
  key)             cmd_key;;
  update)          cmd_update;;
  uninstall)       cmd_uninstall;;
  help|*)          usage;;
esac
