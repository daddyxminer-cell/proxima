#!/usr/bin/env bash
# =============================================================================
#  proxima-gpu-miner.sh — standalone CPU + NVIDIA GPU Proxima miner for a VPS / rig
#
#  Mines on THIS machine and talks to a node that already runs (and is synced)
#  somewhere else.  No node, no database: it builds `proxi` from
#  https://github.com/lunfardo314/proxima (master) and runs `proxi node mine --seeker`
#  with a CUDA nonce seeker (`gpu_seeker`, sources shipped inside this script).
#  Target network: Proxima "take 1" (v1.0.x-take1, miner version 4).
#  Built for proxi v1.0.2-take1 (seeker protocol: one job per round, the seeker keeps improving its output).
#  Note: the key is held in the seeker's memory while it mines - use a dedicated wallet on
#  machines you control (see proxi/node_cmd/mine/nonce_seeker/README.md upstream).
#  Same configuration, prompts, flags and commands as proxima-miner.sh (CPU), plus GPU options.
#
#  GPUs: NVIDIA RTX 20xx / GTX 16xx (Turing, sm_75), RTX 30xx (Ampere, sm_86), RTX 40xx (Ada,
#  sm_89) and RTX 50xx (Blackwell, sm_120) get native code; A100 (80) and H100 (90) are
#  available too (--gpu-arch all).  Anything >= compute 7.5 starts from the embedded PTX.
#  Needs the NVIDIA driver (libcuda; >= 525 for CUDA 12, >= 570 for RTX 50xx).  The CUDA compiler is found on the machine
#  or downloaded privately (no root, nothing installed system-wide).  No NVIDIA GPU
#  or driver?  The script then installs a plain CPU miner.
#
#  `install` asks for what it needs:
#    1. the API URL of your synced node          (default: http://161.97.92.49:8000) - or your own node on this machine
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
#  POOL (several VPS act as ONE miner)  — added in 1.4.0
#    One machine is the SERVER (`install --pool-server`): it runs the miner and a token-protected job server.
#    Every other VPS is a WORKER (`install --pool-worker http://SERVER:8100 --pool-token T --key-file proxima.key`):
#    it installs only the seekers (GPU and/or CPU), no node, no database.  All hashrates add up, the server
#    submits one transit per round, the reward goes to the one key.  `status` on the server lists every member with
#    its hashrate and the pool total; `pool` prints the join command.  Honest limits: every worker holds the wallet
#    key in memory; the token is plain HTTP (firewall the port / tunnel it); the server is a single point of failure;
#    the chance of a win is that of the TOTAL hashrate (the pool does not make blocks easier) and there is no payout
#    sharing between different owners - it is one key.
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
#    node-reset    own-node mode: wipe the local node's database and restore a fresh snapshot
#    logs node     own-node mode: follow the local node's log
#    pool          pool server: members, combined hashrate, the join command;  pool worker: link to the server
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
#                                          normal ~150 ms kernels, the driver-sized resident wave (all SMs busy)
#                                          high   ~300 ms kernels (same wave, fewer launches)
#    --gpu-arch "86 89" [GPU_ARCHS]      CUDA architectures to compile for (default: the
#                                        installed GPUs; "75 86 89 120" when none is visible).
#                                        75 = RTX 20xx / GTX 16xx / T4     86 = RTX 30xx / RTX A-series
#                                        89 = RTX 40xx / L4 / L40          120 = RTX 50xx   (80 A100, 90 H100)
#                                        "all" = 75 80 86 89 90 120 in one build (needs CUDA >= 12.8)
#    --own-node         [OWN_NODE=1]     run YOUR OWN node on this machine (built from source, systemd unit
#                                        proxima-gpu-node, snapshot restore + sync) and let the miner (CPU threads
#                                        + GPU) talk to that node ONLY: no remote node, no failover nodes, no
#                                        extra streams, no witnesses. Implies --no-public. --no-own-node undoes it.
#                                        (the node itself peers with the network on UDP 4000, as any node does)
#    --pool-server      [POOL_ROLE=server]  POOL: this machine is the CENTRE.  Its miner opens a job server (default
#                                        0.0.0.0:8100, protected by a generated token) and every worker VPS that
#                                        joins searches for it: all CPU+GPU hashrate is added up, ONE miner submits,
#                                        ONE key earns.  Prints the exact command to run on each worker.
#                                        --pool-port N  --pool-bind ADDR  --pool-token T  --pool-allow IP,IP (ufw)
#    --pool-worker URL  [POOL_URL]       POOL: this machine only SEARCHES for the pool server at URL (no node, no
#                                        proxi, no database).  Needs --pool-token T and the pool's proxima.key.
#                                        GPU + optional CPU threads (--workers N); --pool-name NAME (default: hostname)
#    --no-pool          [POOL_ROLE=off]  leave the pool: back to a normal stand-alone miner
#    (environment only)  GPU_KERNELS="2 1 0"  which kernel code-size levels to compile (L2 smallest, L0 = fully
#                                        inlined; the seeker times them on each card and keeps the fastest. "2 1"
#                                        skips L0: a much shorter build, and needs less RAM)
#                        GPU_VARIANT=L2      force one kernel instead of timing them (L0 L1 L2, optionally /b4 /b6)
#                        GPU_SEEKER_PIPELINE=0  one launch at a time instead of the queued async path (for comparing)
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

SCRIPT_VERSION="1.4.0"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
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
OWN_OFF=0; [[ "${OWN_NODE:-}" == 0 ]] && OWN_OFF=1
OWN_NODE="${OWN_NODE:-}"               # 1 = run a node of your own on this machine and mine ONLY through it
POOL_OFF=0; [[ "${POOL_ROLE:-}" == off ]] && { POOL_OFF=1; POOL_ROLE=""; }
POOL_ROLE="${POOL_ROLE:-}"             # server | worker | empty (= stand-alone): several VPS searching for ONE miner under ONE key
POOL_PORT="${POOL_PORT:-}"             # server: port of the job server (default 8100)
POOL_BIND="${POOL_BIND:-}"             # server: address it listens on (default 0.0.0.0)
POOL_TOKEN="${POOL_TOKEN:-}"           # shared secret of the pool (server: generated unless given)
POOL_URL="${POOL_URL:-}"               # worker: http://SERVER:PORT of the pool server
POOL_ALLOW="${POOL_ALLOW:-}"           # server: worker IPs to let through ufw
POOL_NAME="${POOL_NAME:-}"             # worker: the name shown on the server's dashboard (default: hostname)
NO_SERVICES="${NO_SERVICES:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
KEY_MODE=""                            # import | generate | keep   (remembered in miner.conf)
SAVED_NODE_URL=""
USE_GPU="${USE_GPU:-}"                 # 1 = mine on the GPU(s), 0 = CPU only, empty = auto (use it if one is found)
GPU_DEVICES="${GPU_DEVICES:-}"         # CUDA device numbers, comma separated; empty = all
GPU_INTENSITY="${GPU_INTENSITY:-}"     # low | normal | high
GPU_VARIANT="${GPU_VARIANT:-}"           # force a kernel level (L0 | L1 | L2 | with /b4 /b6); empty = time them and keep the fastest
GPU_ARCHS="${GPU_ARCHS:-}"             # e.g. "86 89", or "all"; empty = the installed GPUs
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
  NODE_DIR="$MINER_HOME/node"                # own-node mode: proxima.yaml + database of the local node
  NODE_SVC="proxima-gpu-node"
  NODE_UNIT="/etc/systemd/system/$NODE_SVC.service"
  NODE_LOG="$MINER_HOME/node.log"; NODE_PID="$MINER_HOME/node.pid"
  CPU_SEEKER_BIN="$BIN_DIR/pool_cpu_seeker"  # pool worker: the reference Rust nonce seeker (CPU threads)
  CPU_SEEKER_STAMP="$MINER_HOME/cpu-seeker.commit"
  CPU_LOG="$MINER_HOME/cpu-seeker.log"
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
    --own-node)     OWN_NODE=1; shift;;
    --no-own-node)  OWN_NODE=0; OWN_OFF=1; shift;;
    --pool-server)  POOL_ROLE=server; shift;;
    --pool-worker)  POOL_ROLE=worker; POOL_URL="${2:?--pool-worker needs the pool server URL, e.g. http://1.2.3.4:8100}"; shift 2;;
    --pool-port)    POOL_PORT="${2:?--pool-port needs a value}"; shift 2;;
    --pool-bind)    POOL_BIND="${2:?--pool-bind needs a value}"; shift 2;;
    --pool-token)   POOL_TOKEN="${2:?--pool-token needs a value}"; shift 2;;
    --pool-allow)   POOL_ALLOW="${2:?--pool-allow needs a comma list of IPs}"; shift 2;;
    --pool-name)    POOL_NAME="${2:?--pool-name needs a value}"; shift 2;;
    --no-pool)      POOL_OFF=1; POOL_ROLE=""; shift;;
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
norm_flags() {
  if [[ "${NO_PUBLIC:-0}" == 0 ]]; then NO_PUBLIC=""; fi
  if [[ "${OWN_NODE:-0}" == 0 ]]; then OWN_NODE=""; fi
  if [[ -n "$OWN_NODE" ]]; then NO_PUBLIC=1; fi     # own node: the miner talks to nothing else
  return 0
}
norm_flags
conf_get() { grep -E "^$1=" "$CONF_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true; }
load_conf() {
  [[ -f "$CONF_FILE" ]] || return 0
  [[ -n "$MINER_WORKERS" ]] || MINER_WORKERS=$(conf_get MINER_WORKERS)
  [[ -n "$MINER_MAX_KHS" ]] || MINER_MAX_KHS=$(conf_get MINER_MAX_KHS)
  [[ -n "$NO_PUBLIC"     ]] || NO_PUBLIC=$(conf_get NO_PUBLIC)
  [[ -n "$OWN_NODE" || "$OWN_OFF" == 1 ]] || OWN_NODE=$(conf_get OWN_NODE)
  [[ -n "$USE_GPU"       ]] || USE_GPU=$(conf_get USE_GPU)
  [[ -n "$GPU_DEVICES"   ]] || GPU_DEVICES=$(conf_get GPU_DEVICES)
  [[ -n "$GPU_INTENSITY" ]] || GPU_INTENSITY=$(conf_get GPU_INTENSITY)
  [[ -n "$GPU_ARCHS"     ]] || GPU_ARCHS=$(conf_get GPU_ARCHS)
  [[ -n "$KEY_MODE"      ]] || KEY_MODE=$(conf_get KEY_MODE)
  if [[ "$POOL_OFF" != 1 ]]; then
    [[ -n "$POOL_ROLE"  ]] || POOL_ROLE=$(conf_get POOL_ROLE)
    [[ -n "$POOL_PORT"  ]] || POOL_PORT=$(conf_get POOL_PORT)
    [[ -n "$POOL_BIND"  ]] || POOL_BIND=$(conf_get POOL_BIND)
    [[ -n "$POOL_TOKEN" ]] || POOL_TOKEN=$(conf_get POOL_TOKEN)
    [[ -n "$POOL_URL"   ]] || POOL_URL=$(conf_get POOL_URL)
    [[ -n "$POOL_NAME"  ]] || POOL_NAME=$(conf_get POOL_NAME)
  fi
  norm_flags
  return 0
}
save_conf() {
  as_user mkdir -p "$MINER_HOME"
  printf 'MINER_WORKERS=%s\nMINER_MAX_KHS=%s\nNO_PUBLIC=%s\nOWN_NODE=%s\nKEY_MODE=%s\nUSE_GPU=%s\nGPU_DEVICES=%s\nGPU_INTENSITY=%s\nGPU_ARCHS=%s\nPOOL_ROLE=%s\nPOOL_PORT=%s\nPOOL_BIND=%s\nPOOL_TOKEN=%s\nPOOL_URL=%s\nPOOL_NAME=%s\n' \
    "$MINER_WORKERS" "${MINER_MAX_KHS:-0}" "$([[ -n "$OWN_NODE" ]] && echo 0 || echo "${NO_PUBLIC:-0}")" "${OWN_NODE:-0}" "${KEY_MODE:-}" "${USE_GPU:-0}" "${GPU_DEVICES:-}" "${GPU_INTENSITY:-normal}" "${GPU_ARCHS:-}" \
    "${POOL_ROLE:-}" "${POOL_PORT:-}" "${POOL_BIND:-}" "${POOL_TOKEN:-}" "${POOL_URL:-}" "${POOL_NAME:-}" | as_user tee "$CONF_FILE" >/dev/null
  as_user chmod 600 "$CONF_FILE"      # holds the pool token
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
  if [[ "$POOL_ROLE" == worker ]]; then      # a pool worker: no node, no proxi.yaml - just the launcher
    [[ -f "$LAUNCHER" && -n "$POOL_URL" ]] || die "no pool worker installed in $MINER_HOME — run: $0 install --pool-worker http://SERVER:8100 --pool-token T --key-file proxima.key"
    NODE_URL=""; LEDGER_APIS=(); load_passphrase
    return 0
  fi
  [[ -f "$WALLET_DIR/proxi.yaml" ]] || die "nothing installed in $MINER_HOME — run: $0 install"
  NODE_URL=$(profile_node_url); [[ -n "$NODE_URL" ]] || die "no node_url in $WALLET_DIR/proxi.yaml"
  setup_ledger_apis
  load_passphrase
}

# nodes consulted for ledger info: the public ones first (an independent view), then your node
LEDGER_APIS=()
setup_ledger_apis() {
  LEDGER_APIS=()
  # own-node mode: the public nodes are only asked read-only questions here (is my node in step? same ledger?);
  # the miner process itself never connects to them
  [[ -n "$NO_PUBLIC" && -z "$OWN_NODE" ]] || LEDGER_APIS+=("${PUBLIC_APIS[@]}")
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
  [[ -z "$NO_PUBLIC" || -n "$OWN_NODE" ]] || return 0
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
    warn "pin a matching tag or commit with --branch <tag|commit> (v1.0.2-take1 = commit 60515d0)."
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
  # a longer API timeout: the default is short, and a slow or distant node then looks "not answering"
  if ! grep -qE '^\s+timeout_sec:' "$y"; then
    local ind2; ind2=$(grep -E '^\s+node_url:' "$y" | head -1 | sed -E 's/^(\s+).*/\1/' || true)
    if [[ -n "$ind2" ]]; then
      as_user sed -i -E "/^\s+node_url:/a\\${ind2}timeout_sec: 30" "$y"
      ok "wallet profile: api.timeout_sec 30"
    fi
  fi
  if [[ -z "$NO_PUBLIC" ]] && ! grep -qE '^\s+node_urls:' "$y"; then
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
    if [[ -n "$OWN_NODE" ]]; then log "own-node mode: no witness nodes in the profile — the wallet talks to your own node only"
    else warn "--no-public: no witness nodes in the profile — the node's ledger library is NOT cross-checked"; fi
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
  pool_apply_profile
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
    *"B200"*|*"B100"*|*"GB200"*) echo 10.0;;
    *"RTX 50"[0-9][0-9]*|*"RTX PRO"*"Blackwell"*) echo 12.0;;
    *"H100"*|*"H200"*|*"H800"*|*"H20"|*"H20 "*|*"GH200"*) echo 9.0;;
    *"RTX 40"[0-9][0-9]*|*"RTX 6000 Ada"*|*"RTX 5000 Ada"*|*"RTX 4500 Ada"*|*"RTX 4000 Ada"*|*"RTX 2000 Ada"*|*"L40"*|*"L4"|*"L4 "*) echo 8.9;;
    *"A100"*|*"A800"*|*"A30"|*"A30 "*) echo 8.0;;                                    # before A10: "A100" contains "A10"
    *"RTX 30"[0-9][0-9]*|*"RTX A"[0-9]*|*"A40"*|*"A10"*|*"A16"*|*"A2"|*"A2 "*) echo 8.6;;
    *"RTX 20"[0-9][0-9]*|*"GTX 16"[0-9][0-9]*|*"TITAN RTX"*|*"Titan RTX"*|*"Quadro RTX"*|*"T4"*|*"T1000"*|*"T600"*|*"T400"*) echo 7.5;;
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
  # "all" = every supported card: RTX 20xx/GTX 16xx (75), A100 (80), RTX 30xx (86), RTX 40xx (89), H100 (90), RTX 50xx (120)
  if [[ "$GPU_ARCHS" == all ]]; then echo "75 80 86 89 90 120"; return 0; fi
  if [[ -n "$GPU_ARCHS" ]]; then tr ',' ' ' <<<"$GPU_ARCHS" | xargs; return 0; fi
  for cc in ${GPU_CCS[@]+"${GPU_CCS[@]}"}; do [[ -n "$cc" ]] && list+=("$(arch_of_cc "$cc")"); done
  if (( ${#list[@]} == 0 )); then echo "75 86 89 120"; return 0; fi   # nothing detected (building elsewhere): the mainstream cards
  for a in "${list[@]}"; do echo "$a"; done | sort -un | xargs
}
check_archs() {
  local a
  for a in $(gpu_arch_list); do
    [[ "$a" =~ ^[0-9]{2,3}$ ]] && (( a >= 75 )) || die "bad CUDA architecture '$a' (--gpu-arch \"86 89\" or --gpu-arch all; the kernel needs compute capability >= 7.5: RTX 20xx/30xx/40xx/50xx)"
  done
}
need_cuda_ver() { local a max=0; for a in $(gpu_arch_list); do (( a > max )) && max=$a; done; if (( max >= 100 )); then echo 12.8; else echo 12.0; fi; }  # sm_120 (RTX 50xx) needs nvcc >= 12.8

gpu_params() { # GPU_INTENSITY -> kernel length / queue depth (the seeker adapts the real launch size to the target)
  case "${GPU_INTENSITY:-normal}" in
    low)    GPU_TARGET_MS=30;  GPU_BPS=1;;     # one block per SM: the card stays responsive for the desktop
    high)   GPU_TARGET_MS=300; GPU_BPS=;;      # BPS empty = the seeker asks the driver for the exact resident wave
    *)      GPU_TARGET_MS=150; GPU_BPS=;;
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
seeker_installed() { [[ -x "$SEEKER_BIN" ]] && compgen -G "${SEEKER_FATBIN%.fatbin}*.fatbin" >/dev/null; }
gpu_stale() { # 0 = the seeker has to be (re)built
  [[ -x "$SEEKER_BIN" && -f "$GPU_INFO" ]] || return 0
  [[ -f "$SEEKER_FATBIN" || -f "${SEEKER_FATBIN%.fatbin}.l2.fatbin" || -f "${SEEKER_FATBIN%.fatbin}.l1.fatbin" ]] || return 0
  [[ "$(head -n1 "$GPU_INFO" 2>/dev/null)" == "$(gpu_build_key)" ]] && return 1
  return 0
}

build_gpu_seeker() {
  step "GPU seeker: CUDA kernel + Rust host"
  [[ -d "$SRC_DIR/util/vrf/rust" ]] || die "proxima source not found in $SRC_DIR (fetch_source first)"
  check_archs
  unpack_gpu_src
  as_user mkdir -p "$GPU_BUILD" "$GPU_TARGET"
  local archs a min=999 gen=() ccb; archs=$(gpu_arch_list)
  for a in $archs; do gen+=(-gencode "arch=compute_$a,code=sm_$a"); (( a < min )) && min=$a; done
  # PTX of the LOWEST architecture: a card that has no SASS in the list (say a 3060 when only 89 was asked for)
  # or a future one still starts, the driver compiles the PTX once (slower first start, same speed after)
  gen+=(-gencode "arch=compute_$min,code=compute_$min")
  ccb=$(host_ccbin)
  local kb; kb=$(awk '/^(MemAvailable|SwapFree):/{s+=$2} END{print s+0}' /proc/meminfo 2>/dev/null || echo 0)
  if (( kb > 0 && kb < 1700000 )); then
    warn "only $((kb / 1024)) MB of free RAM+swap: nvcc needs about 1 GB per architecture for the kernel and may be OOM-killed — add swap if it fails:"
    warn "  fallocate -l 3G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
  fi
  # Kernel code-size levels (see gpu-seeker/cuda/vrf_core.h): the fully inlined kernel (L0) is 0.7-1.5 MB of machine code,
  # far more than a GPU's instruction cache, so the cores stall on fetching code (100% "utilisation" but only a third of the
  # power limit). L2 is ~160 KB, L1 ~260 KB. The seeker times every level on every card at start-up and keeps the fastest.
  local lv lvs="${GPU_KERNELS:-2 1 0}" suffix built=()
  log "nvcc: sm_${archs// /, sm_} + PTX compute_$min, kernel levels: $lvs${ccb:+  (host compiler $ccb)}  — 1 to 5 minutes"
  for lv in $lvs; do
    [[ "$lv" =~ ^[012]$ ]] || { warn "ignoring unknown kernel level '$lv' in GPU_KERNELS"; continue; }
    suffix=""; [[ "$lv" != 0 ]] && suffix=".l$lv"
    log "  kernel level L$lv …"
    if ! ( cd "$GPU_SRC/cuda" && as_user env PATH="$(dirname "$NVCC"):$PATH" "$NVCC" -U_GNU_SOURCE -O3 -std=c++14 --fatbin -DVRF_CALL_LEVEL="$lv" \
             "${gen[@]}" ${ccb:+-ccbin "$ccb"} -o "$GPU_BUILD/gpu_seeker$suffix.fatbin.new" kernel.cu ); then
      if (( ${#built[@]} > 0 )); then warn "level L$lv failed to compile — continuing with the levels that did (${built[*]})"; continue; fi
      die "nvcc failed. Typical causes: a host gcc newer than 14 (install g++-13 or pass GPU_CCBIN=/usr/bin/g++-13), out of memory (add swap), or an architecture this nvcc does not know (--gpu-arch \"86 89\"; sm_120 needs CUDA >= 12.8)"
    fi
    built+=("L$lv")
  done
  (( ${#built[@]} > 0 )) || die "no kernel level was built (GPU_KERNELS='$lvs')"
  ok "kernels compiled: ${built[*]} ($(cat "$GPU_BUILD"/gpu_seeker*.fatbin.new | wc -c | awk '{printf "%d", $1/1024}') KB)"
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
  # remove the images of a previous build first, so a level that was left out this time cannot linger
  as_user rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin" 2>/dev/null || as_root rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin"
  local f dst
  for f in "$GPU_BUILD"/gpu_seeker*.fatbin.new; do
    [[ -f "$f" ]] || continue
    dst="$BIN_DIR/$(basename "${f%.new}")"
    put_file "$f" "$dst" 0644
  done
  as_user rm -f "$GPU_BUILD/gpu_seeker.new" "$GPU_BUILD"/gpu_seeker*.fatbin.new
  # `proxi node mine --seeker` looks for a program called nonce_seeker on PATH: the launcher puts this directory first
  as_user mkdir -p "$SEEKER_PATH_DIR"; as_user ln -sf "$SEEKER_BIN" "$SEEKER_PATH_DIR/nonce_seeker"
  printf '%s\nnvcc=%s\n' "$(gpu_build_key)" "$(nvcc_version "${NVCC:-nvcc}" 2>/dev/null || true)" | as_user tee "$GPU_INFO" >/dev/null
  ok "gpu_seeker + gpu_seeker.fatbin -> $BIN_DIR"
}

# environment the seeker runs with: the launcher exports the same
seeker_env() {
  gpu_params
  SEEKER_ENV=(GPU_SEEKER_FATBIN="$SEEKER_FATBIN" GPU_SEEKER_TARGET_MS="$GPU_TARGET_MS")
  if [[ -n "$GPU_BPS" ]]; then SEEKER_ENV+=(GPU_SEEKER_BLOCKS_PER_SM="$GPU_BPS"); fi
  if [[ -n "$GPU_VARIANT" ]]; then SEEKER_ENV+=(GPU_SEEKER_VARIANT="$GPU_VARIANT"); fi
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
  local i cc n old=0 need50=0 fam=""
  for i in "${!GPU_CCS[@]}"; do
    cc="${GPU_CCS[$i]}"; n="${GPU_NAMES[$i]}"
    [[ -n "$cc" ]] || { warn "GPU ${GPU_IDX[$i]} ($n): compute capability unknown — the kernel is written for >= 7.5 (RTX 20xx and newer); try --gpu-arch all"; continue; }
    if   awk -v c="$cc" 'BEGIN{exit !(c >= 10)}';  then need50=1; fam+=" RTX-50xx/Blackwell(sm_${cc/./})"
    elif awk -v c="$cc" 'BEGIN{exit !(c >= 8.9)}'; then fam+=" RTX-40xx/Ada(sm_${cc/./})"
    elif awk -v c="$cc" 'BEGIN{exit !(c >= 8.6)}'; then fam+=" RTX-30xx/Ampere(sm_${cc/./})"
    elif awk -v c="$cc" 'BEGIN{exit !(c >= 8.0)}'; then fam+=" Ampere-datacenter(sm_${cc/./})"
    elif awk -v c="$cc" 'BEGIN{exit !(c >= 7.5)}'; then fam+=" RTX-20xx/Turing(sm_${cc/./})"
    else old=1; fi
  done
  [[ -z "$fam" ]] || log "kernel targets:$fam"
  if (( old )); then warn "a GPU older than compute 7.5 (before RTX 20xx / GTX 16xx: Pascal, Volta) is not supported by the kernel"; fi
  if [[ -n "$GPU_DRIVER" ]]; then
    # the kernels are built with CUDA 12.x: its minor-version compatibility needs driver >= 525 (Linux)
    if ! ver_ge "$GPU_DRIVER" 525; then
      warn "NVIDIA driver $GPU_DRIVER is older than 525: the CUDA 12 kernel cannot load (RTX 20xx/30xx included) — update the driver (>= 535 is a good choice)"
    fi
    if (( need50 )) && ! ver_ge "$GPU_DRIVER" 570; then
      warn "RTX 50xx (Blackwell) needs NVIDIA driver >= 570; this machine has $GPU_DRIVER — update the driver or the GPU seeker will not start"
    fi
  fi
  return 0
}

# choose_gpu: decides USE_GPU / GPU_DEVICES / GPU_INTENSITY (asks unless flags or saved settings decide)
choose_gpu() {
  step "GPU"
  detect_gpus
  if (( GPU_N == 0 )); then
    if [[ "$USE_GPU" == 1 ]]; then
      die "--gpu given but no NVIDIA GPU answers nvidia-smi${GPU_PCI_NAME:+ (lspci sees: $GPU_PCI_NAME — the driver is missing or not loaded)}. Install the NVIDIA driver (>= 535; >= 570 for RTX 50xx), reboot, then retry."
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
    def=0
    # own node + CPU + GPU: the node needs about a core and a half, the GPU feeder hardly anything - the rest mines
    if [[ -n "$OWN_NODE" ]]; then def=$(( n > 3 ? n - 2 : 1 )); fi
    if (( SAVED_HAD_GPU )) && [[ "$SAVED_WORKERS" =~ ^[0-9]+$ ]]; then def="$SAVED_WORKERS"; fi
    if [[ -z "$MINER_WORKERS" ]]; then ask MINER_WORKERS "extra CPU threads next to the GPU (this machine has $n cores; 0 = GPU only)" "$def"; fi
    [[ "$MINER_WORKERS" =~ ^[0-9]+$ ]] || die "invalid thread count: '$MINER_WORKERS'"
  else
    if [[ -z "$MINER_WORKERS" ]]; then ask MINER_WORKERS "CPU threads to mine with (this machine has $n)" "$([[ "$SAVED_WORKERS" =~ ^[1-9][0-9]*$ ]] && echo "$SAVED_WORKERS" || echo "$n")"; fi
    if [[ "$POOL_ROLE" == server ]]; then
      [[ "$MINER_WORKERS" =~ ^[0-9]+$ ]] || die "invalid thread count: '$MINER_WORKERS'"
      (( MINER_WORKERS >= 1 )) || log "0 CPU threads here: this server only coordinates, the workers do all the searching"
    else
      [[ "$MINER_WORKERS" =~ ^[0-9]+$ && "$MINER_WORKERS" -ge 1 ]] || die "invalid thread count: '$MINER_WORKERS' (at least 1 without a GPU)"
    fi
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
#  Own node (--own-node): a Proxima node on THIS machine; the miner talks to it and to nothing else
# =============================================================================
# The node itself still peers with the public network (it has to, to receive and publish transactions):
# "mines to itself" means the MINER connects only to the local node: no other API, no extra streams, no failover.
node_api_port() {
  local p; p=$(awk '/^api:/{f=1;next} f&&/^[^ #]/{f=0} f&&/^ +port:/{print $2;exit}' "$NODE_DIR/proxima.yaml" 2>/dev/null || true)
  echo "${p:-8000}"
}
node_peer_port() {
  local p; p=$(awk '/^  host:/{f=1;next} f&&/^    port:/{print $2;exit}' "$NODE_DIR/proxima.yaml" 2>/dev/null || true)
  echo "${p:-4000}"
}
own_node_url() { echo "http://127.0.0.1:$(node_api_port)"; }
node_managed() { [[ -f "$NODE_DIR/proxima.yaml" ]]; }     # a node set up by this script
node_unit_installed() { have_systemd && [[ -f "$NODE_UNIT" ]]; }

build_node_binary() {
  step "Compiling the node (proxima $SRC_VERSION)"
  as_user mkdir -p "$BUILD_OUT"
  local par=(); if low_mem; then par=(-p 1); fi
  ( cd "$SRC_DIR" && log "proxima (node) …" && go_run build ${par[@]+"${par[@]}"} -trimpath -o "$BUILD_OUT/proxima.new" . )
  local v; v=$("$BUILD_OUT/proxima.new" version 2>/dev/null | awk '/Version:/{print $2}' || true)
  [[ "$v" == "$SRC_VERSION" ]] || die "built node reports '${v:-nothing}', expected $SRC_VERSION"
  ok "proxima (node) $v"
}
install_node_binary() {
  local dst="$BIN_DIR/proxima"
  [[ -f "$BUILD_OUT/proxima.new" ]] || return 0
  as_user mkdir -p "$BIN_DIR" 2>/dev/null || as_root mkdir -p "$BIN_DIR"
  if [[ -f "$dst" ]]; then { as_user cp -f "$dst" "$dst.prev" 2>/dev/null || as_root cp -f "$dst" "$dst.prev"; } || true; fi
  { as_user install -m 0755 "$BUILD_OUT/proxima.new" "$dst.tmp" 2>/dev/null || as_root install -m 0755 "$BUILD_OUT/proxima.new" "$dst.tmp"; }
  { as_user mv -f "$dst.tmp" "$dst" 2>/dev/null || as_root mv -f "$dst.tmp" "$dst"; }
  rm -f "$BUILD_OUT/proxima.new"
  ok "proxima (node) -> $dst"
}

# the public nodes' libp2p host IDs change when the testnet is reset: refresh the static peer entries
refresh_public_peers() {
  local yaml="$NODE_DIR/proxima.yaml"
  [[ -f "$yaml" ]] || return 0
  local u ip id old n=0
  for u in "${PUBLIC_APIS[@]}"; do
    ip=${u#http://}; ip=${ip%%:*}
    id=$(curl -fs -m 8 "$u/api/v1/node_info" 2>/dev/null | jq -r '.id // empty' 2>/dev/null || true)
    [[ "$id" =~ ^12D3KooW[A-Za-z0-9]+$ ]] || continue
    old=$(grep -oE "/ip4/$ip/udp/[0-9]+/quic-v1/p2p/[A-Za-z0-9]+" "$yaml" 2>/dev/null | head -1 | awk -F/p2p/ '{print $2}' || true)
    [[ -n "$old" && "$old" != "$id" ]] || continue
    as_user sed -i -E "s#(/ip4/$ip/udp/[0-9]+/quic-v1/p2p/)[A-Za-z0-9]+#\1$id#" "$yaml"
    n=$((n + 1)); log "peer $ip: host ID $old -> $id"
  done
  if (( n > 0 )); then ok "refreshed $n public peer ID(s) in proxima.yaml"; fi
  return 0
}

configure_own_node() {
  step "Own node: configuration ($NODE_DIR)"
  as_user mkdir -p "$NODE_DIR"
  if [[ -f "$NODE_DIR/proxima.yaml" ]]; then
    ok "keeping the existing proxima.yaml (host ID $(awk '/^    id:/{print $2;exit}' "$NODE_DIR/proxima.yaml"))"
    refresh_public_peers
  else
    ( cd "$NODE_DIR" && as_user "$BIN_DIR/proxi" config node -v >/dev/null ) || die "proxi config node failed"
    as_user chmod 600 "$NODE_DIR/proxima.yaml"
    ok "proxima.yaml created (public testnet peers + snapshot/sync sources pre-filled)"
  fi
  local pport aport; pport=$(node_peer_port); aport=$(node_api_port)
  # a node (this script's or another one) that already holds the ports would make ours fail to start
  if ! node_unit_installed && [[ -z "$(node_pid)" ]] && curl -fs -m 3 "http://127.0.0.1:$aport/api/v1/sync_info" >/dev/null 2>&1; then
    warn "something already answers on 127.0.0.1:$aport (another Proxima node on this machine?). Stop it, or change api.port in $NODE_DIR/proxima.yaml"
  fi
  if command -v ufw >/dev/null && as_root ufw status 2>/dev/null | grep -q "Status: active"; then
    as_root ufw allow "$pport/udp" >/dev/null && ok "ufw: opened $pport/udp (peering)"
  else
    log "make sure UDP $pport (peering, QUIC) is reachable from the internet, otherwise the node only has outgoing peers"
  fi
  NODE_URL="http://127.0.0.1:$aport"
}

write_node_unit() {
  have_systemd || return 0
  as_root tee "$NODE_UNIT" >/dev/null <<UNIT_END
[Unit]
Description=Proxima node of the CPU+GPU miner ($NODE_SVC)
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
UNIT_END
  as_root systemctl daemon-reload
  ok "$NODE_SVC.service"
}

node_pid() {
  local p=""
  if node_unit_installed; then p=$(systemctl show -p MainPID --value "$NODE_SVC" 2>/dev/null || true)
  elif [[ -f "$NODE_PID" ]]; then p=$(cat "$NODE_PID" 2>/dev/null || true); fi
  if [[ ! "$p" =~ ^[0-9]+$ ]] || ! kill -0 "$p" 2>/dev/null; then p=$(pgrep -n -f "^$BIN_DIR/proxima( |\$)" 2>/dev/null || true); fi
  if [[ "$p" =~ ^[0-9]+$ && "$p" != 0 ]]; then echo "$p"; fi
  return 0
}
node_state() { # active|inactive|running|stopped|none
  node_managed || { echo none; return 0; }
  if node_unit_installed; then systemctl is-active "$NODE_SVC" 2>/dev/null || true; return 0; fi
  if [[ -n "$(node_pid)" ]]; then echo running; else echo stopped; fi
}
node_svc() { # node_svc start|stop|restart|enable
  local action="$1" pid i
  node_managed || return 0
  if node_unit_installed; then
    case "$action" in
      enable) as_root systemctl enable "$NODE_SVC" >/dev/null 2>&1; as_root systemctl restart "$NODE_SVC"; ok "enabled + started $NODE_SVC";;
      *)      as_root systemctl "$action" "$NODE_SVC"; ok "$action $NODE_SVC";;
    esac
    return 0
  fi
  case "$action" in
    stop|restart)
      pid=$(node_pid)
      if [[ -n "$pid" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done   # the node flushes its database
        if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
        ok "stopped the node (pid $pid)"
      fi
      rm -f "$NODE_PID";;
  esac
  case "$action" in
    start|restart|enable)
      if [[ -n "$(node_pid)" ]]; then ok "node already running (pid $(node_pid))"; return 0; fi
      spawn_detached "$NODE_DIR" "$NODE_LOG" "$NODE_PID" "$BIN_DIR/proxima"
      ok "node started in the background (pid $(cat "$NODE_PID" 2>/dev/null || echo ?), log $NODE_LOG)";;
  esac
  return 0
}

# wait for the local node to restore a snapshot and catch up (first start: a few minutes)
wait_own_node_sync() {
  step "Own node: waiting for the snapshot restore and sync (Ctrl-C to skip: the miner waits for it anyway)"
  local url s i cur lrb
  url=$(own_node_url)
  for i in $(seq 1 180); do
    s=$(sync_info "$url")
    if [[ -n "$s" ]]; then
      if grep -q '"synced": *true' <<<"$s"; then echo; ok "own node is synced: $(tr -d '\n ' <<<"$s")"; return 0; fi
      cur=$(jq -r .current_slot <<<"$s" 2>/dev/null || true); lrb=$(jq -r .lrb_slot <<<"$s" 2>/dev/null || true)
      printf '\r  … syncing: LRB slot %s, current %s (%s behind)      ' "${lrb:-?}" "${cur:-?}" "$(( ${cur:-0} - ${lrb:-0} ))"
    else printf '\r  … node API not up yet / restoring the snapshot (%ds)   ' $((i * 10)); fi
    sleep 10
  done
  echo; warn "the node is not synced after 30 min — see: $0 logs node   (the miner starts when it is)"
  return 0
}

# the DB of a node that belongs to an old ledger cannot sync: `node-reset` wipes it and restores a fresh snapshot
wipe_own_node_db() {
  as_user rm -rf "$NODE_DIR/proximadb" "$NODE_DIR/proximadb.txstore" "$NODE_DIR/proximadb.txlog"
  as_user bash -c 'rm -f "$1"/*.snapshot' _ "$NODE_DIR"
  ok "node database wiped: it restores the network's latest snapshot on the next start"
}
cmd_node_reset() {
  load_install
  [[ -n "$OWN_NODE" ]] && node_managed || die "no own node here (install with --own-node)"
  confirm "Stop the node, delete its database and restore from the network's latest snapshot?" y || exit 0
  svc stop 2>/dev/null || true
  node_svc stop
  wipe_own_node_db
  node_svc start
  svc start
  wait_own_node_sync
}

# =============================================================================
#  Launcher + service
# =============================================================================
# One small script is what systemd (or the background fallback) runs. It waits until your node
# is synced, then execs `proxi node mine`. The node URL is read from proxi.yaml at every start,
# so `set-node` only has to edit the profile.
write_launcher() {
  if [[ "$POOL_ROLE" == worker ]]; then write_launcher_worker; return 0; fi
  local streams="" fallbacks="" own=0
  if [[ -n "$OWN_NODE" ]]; then own=1                 # own node: no extra streams, no fallback nodes - nothing but the local node
  else
    [[ -n "$NO_PUBLIC" ]] || streams=$(IFS=,; echo "${PUBLIC_APIS[*]}")
    fallbacks="$streams"
  fi
  local body
  body=$(
    printf '#!/usr/bin/env bash\n# generated by proxima-gpu-miner.sh %s — regenerated on install/update/restart-after-change\n' "$SCRIPT_VERSION"
    printf 'WALLET_DIR=%q\nENV_FILE=%q\nPROXI=%q\nWORKERS=%q\nMAX_KHS=%q\nSTREAMS=%q\nUSE_GPU=%q\nSEEKER_PATH_DIR=%q\nFALLBACKS=%q\nOWN=%q\n' \
      "$WALLET_DIR" "$ENV_FILE" "$BIN_DIR/proxi" "$MINER_WORKERS" "${MINER_MAX_KHS:-0}" "$streams" "${USE_GPU:-0}" "$SEEKER_PATH_DIR" "$fallbacks" "$own"
    if [[ "${USE_GPU:-0}" == 1 ]]; then
      seeker_env
      # proxi spawns its seeker with a fixed command line: gpu_seeker is tuned through these variables
      local e; for e in "${SEEKER_ENV[@]}"; do printf 'export %q\n' "$e"; done
    fi
    cat <<'EOT'
cd "$WALLET_DIR" || exit 1
if [ -r "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
primary=$(awk '/^[ \t]+(node_url|endpoint):/{print $2;exit}' proxi.yaml)
[ -n "$primary" ] || { echo "no node_url in $WALLET_DIR/proxi.yaml"; exit 1; }

# ---- node selection with failover ---------------------------------------------------------------
# One unreachable node used to stop the miner ("list the sequencers failed: node is not answering").
# Candidates, in order: your node (proxi.yaml), every URL in $WALLET_DIR/nodes.list (optional, one per
# line), then the public nodes. The first one that answers AND is synced wins; your own node is
# preferred again as soon as it is back.
CANDS=("$primary")
if [ "$OWN" != 1 ] && [ -r "$WALLET_DIR/nodes.list" ]; then
  while read -r l; do case "$l" in http*) CANDS+=("$l");; esac; done <"$WALLET_DIR/nodes.list"
fi
IFS=, read -r -a _pub <<<"$FALLBACKS"
for l in "${_pub[@]}"; do [ -n "$l" ] && CANDS+=("$l"); done

probe() { curl -fs -m 6 "$1/api/v1/sync_info" 2>/dev/null; }          # prints the JSON; rc != 0 when down
is_synced() { grep -q '"synced": *true'; }
pick() {   # sets CHOSEN (first synced node, else the first reachable one); rc 1 when nothing answers
  local u j reach=""
  CHOSEN=""; SYNCED=1
  for u in "${CANDS[@]}"; do
    j=$(probe "$u") || continue
    if is_synced <<<"$j"; then CHOSEN="$u"; return 0; fi
    [ -n "$reach" ] || reach="$u"
  done
  [ -n "$reach" ] || return 1
  CHOSEN="$reach"; SYNCED=0; return 0
}

i=0
while :; do
  if pick; then
    [ "$SYNCED" = 1 ] && break
    # reachable but behind: give it ~10 min, then mine on it anyway (own node: wait for as long as it takes - there is no other node)
    if [ "$OWN" != 1 ] && [ "$i" -ge 60 ]; then echo "no node is synced after 10 min - starting on $CHOSEN anyway"; break; fi
  fi
  if [ "$OWN" = 1 ]; then [ $((i % 6)) -eq 0 ] && echo "waiting for your own node $primary to be up and synced (it restores a snapshot on its first start)..."
  else [ $((i % 6)) -eq 0 ] && echo "waiting for a reachable, synced node ($primary, then fallbacks)..."; fi
  i=$((i + 1)); sleep 10
done
if [ "$OWN" = 1 ]; then echo "own-node mode: mining ONLY through $CHOSEN (no other node, no extra streams)"
elif [ "$CHOSEN" = "$primary" ]; then echo "node $CHOSEN is synced"
else echo "WARNING: your node $primary is not usable - mining through $CHOSEN until it is back"; fi

# -v: one "window ... at N H/s" line per round. Without it the miner prints a rate only when a round lasts >2 s,
# which at low difficulty is almost never (the dashboard then has no hashrate to show)
args=(node mine -v --workers "$WORKERS" --api.node_url "$CHOSEN")
if [ "$USE_GPU" = 1 ]; then
  # proxi starts the program called `nonce_seeker` (found first in SEEKER_PATH_DIR: it is gpu_seeker),
  # hands it the job server address, token and key, restarts it if it dies and stops it with the miner
  export PATH="$SEEKER_PATH_DIR:$PATH"
  args+=(--seeker)
fi
[ "$MAX_KHS" != 0 ] && args+=(--max-hashrate-khs "$MAX_KHS")
[ -n "$STREAMS" ] && args+=(--stream "$STREAMS")

# ---- watchdog -------------------------------------------------------------------------------------
# proxi runs in the background; every 20 s we look at the node it uses:
#  * down for ~80 s                                      -> stop the miner and re-run this script, which
#                                                           picks another node
#  * on a fallback and the primary is back and synced    -> same, to switch back
"$PROXI" "${args[@]}" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 0' TERM INT
bad=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 20 & wait $! || true
  kill -0 "$pid" 2>/dev/null || break
  if probe "$CHOSEN" >/dev/null; then bad=0; else bad=$((bad + 1)); fi
  if [ "$bad" -ge 4 ]; then echo "node $CHOSEN unreachable for ~80 s - restarting the miner on another node"; break; fi
  if [ "$CHOSEN" != "$primary" ] && j=$(probe "$primary") && is_synced <<<"$j"; then
    echo "your node $primary is back and synced - switching back to it"; break
  fi
done
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 3; exec "$0" "$@"      # same PID, so systemd and the background fallback both keep supervising it
fi
wait "$pid"; exit $?
EOT
  )
  printf '%s\n' "$body" | as_user tee "$LAUNCHER" >/dev/null
  as_user chmod 700 "$LAUNCHER"
  ok "launcher: $LAUNCHER"
}

write_unit() {
  have_systemd || { warn "systemd not available (or --no-services): the miner will run as a background process"; return 0; }
  local after="network-online.target time-sync.target" wants="network-online.target"
  if [[ -n "$OWN_NODE" ]]; then after+=" $NODE_SVC.service"; wants+=" $NODE_SVC.service"; fi
  as_root tee "$UNIT_FILE" >/dev/null <<EOF
[Unit]
Description=Proxima CPU+GPU $([[ "$POOL_ROLE" == worker ]] && echo "pool worker (seeker -> $POOL_URL)" || echo "miner (proxi node mine -> $MINER_SVC)")
After=$after
Wants=$wants

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
LimitNOFILE=65536

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
  local title="  PROXIMA CPU+GPU MINER" stamp
  if [[ "$POOL_ROLE" == server ]]; then title="  PROXIMA POOL SERVER"; elif [[ "$POOL_ROLE" == worker ]]; then title="  PROXIMA POOL WORKER"; fi; stamp="$(date -u '+%F %T') UTC  "
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
  if [[ -n "$OWN_NODE" ]]; then sect "YOUR OWN NODE (mining only through it)"; else sect "YOUR NODE"; fi
  kv "API" "${C_W}$NODE_URL${C_0}"
  if [[ -n "$OWN_NODE" ]]; then
    local nst npid; nst=$(node_state); npid=$(node_pid)
    kv "node service" "$(svc_badge "$nst")   ${C_D}$(proc_stats "$npid")${C_0}"
    kv "miner talks to" "${C_G}only this node${C_0} ${C_D}(no fallback nodes, no extra streams)${C_0}"
    if [[ "$nst" == none || "$nst" == stopped || "$nst" == inactive ]]; then note "the node is not running: $0 start   (log: $0 logs node)"; fi
  fi
  if ! probe_node "$NODE_URL"; then
    kv "link" "$(dot_off) ${C_R}not reachable${C_0}   ${C_D}(node down, or its firewall blocks this VPS)${C_0}"
    if [[ -n "$OWN_NODE" ]]; then note "the miner waits for it; follow the node log with:  $0 logs node"; else note "the miner keeps retrying; check:  $0 check"; fi
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
    printf '  %s%s ✘ this node is on ANOTHER ledger (genesis %s) — it needs: %s %s\n' "$C_B" "$C_R" "$NP_GEN" "$(if [[ -n "$OWN_NODE" ]]; then echo "$0 node-reset"; else echo "proxima-setup.sh reset"; fi)" "$C_0"
  fi
  if [[ -n "$OWN_NODE" ]]; then
    local pi np na; pi=$(curl -fs -m 6 "$NODE_URL/api/v1/peers_info" 2>/dev/null || true)
    if [[ -n "$pi" ]] && jq -e '.peers' >/dev/null 2>&1 <<<"$pi"; then
      np=$(jq '(.peers // []) | length' <<<"$pi" 2>/dev/null || echo 0); na=$(jq '[(.peers // [])[] | select(.is_alive)] | length' <<<"$pi" 2>/dev/null || echo 0)
      if (( ${na:-0} > 0 )); then kv "peers" "${C_G}${na}${C_0} ${C_D}alive of ${np} known${C_0}"
      else kv "peers" "${C_R}none alive${C_0} ${C_D}(UDP $(node_peer_port) outgoing blocked? check the firewall)${C_0}"; fi
    fi
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
  local rate="" samples="" tot="" k="" solved=0 improved=0 err="" tslot="" lastn="" failover=""
  local key val
  while IFS=$'\t' read -r key val; do
    case "$key" in rate) rate=$val;; samples) samples=$val;; tot) tot=$val;; k) k=$val;; solved) solved=$val;;
                   improved) improved=$val;; err) err=$val;; tslot) tslot=$val;; lastn) lastn=$val;; failover) failover=$val;; esac
  done < <(awk '
    { if (match($0,/mining\.\.\. [0-9_]+ attempts +[0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,f," "); r=f[4]; gsub(/_/,"",r); sm[++ns]=r }
      if (match($0,/attempts at [0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,f," "); r=f[3]; gsub(/_/,"",r); if (r+0>0) sm[++ns]=r }
      if (index($0,"totals:")) tot=substr($0,index($0,"totals:"))
      if (match($0,/difficulty K=[0-9]+/)) k=substr($0,RSTART+13,RLENGTH-13)
      if (match($0,/target slot [0-9]+/)) ts=substr($0,RSTART+12,RLENGTH-12)
      if (match($0,/mining transit #[0-9]+/)) ln=substr($0,RSTART+16,RLENGTH-16)
      if (index($0,"SOLVED transit")) sv++
      if (index($0,"IMPROVED transit")) im++
      if (index($0,"submit failed")||index($0,"not answering")||index($0,"Error:")) er=$0
      if (index($0,"node is answering again")||index($0,"switching back to it")) er=""
      if (index($0,"is not usable - mining through")||index($0,"restarting the miner on another node")) fo=$0
      if (index($0,"no solution for slot")) ns2++ }
    END { if (ns>0) { print "rate\t" sm[ns]; s=""; st=(ns>40)?ns-39:1; for(i=st;i<=ns;i++) s=s sm[i] " "; print "samples\t" s }
          print "tot\t" tot; print "k\t" k; print "solved\t" sv+0; print "improved\t" im+0; print "err\t" er; print "tslot\t" ts; print "lastn\t" ln; print "failover\t" fo }' <<<"$log") || true

  if [[ "$rate" =~ ^[0-9]+$ ]]; then
    local avg peak; read -r avg peak < <(awk '{n=0;s=0;m=0; for(i=1;i<=NF;i++){ if(NF>5 && i<=2) continue; s+=$i;n++; if($i>m)m=$i } printf "%d %d", (n?s/n:0), m}' <<<"$samples") || true
    kv "hashrate" "${C_B}${C_G}$(fmt_rate "$rate")${C_0} ${C_D}$(if [[ "$POOL_ROLE" == server ]]; then echo "WHOLE POOL (this machine + workers)"; elif [[ "${USE_GPU:-0}" == 1 ]]; then echo "CPU+GPU total"; fi)${C_0}   ${C_D}avg${C_0} ${C_W}$(fmt_rate "${avg:-0}")${C_0}  ${C_D}peak${C_0} ${C_W}$(fmt_rate "${peak:-0}")${C_0}"
    local sp; sp=$(tr ' ' '\n' <<<"$samples" | awk 'NF' | spark)
    [[ -n "$sp" ]] && kv "recent" "${C_C}$sp${C_0}"
  else
    kv "hashrate" "${C_Y}no samples yet${C_0} ${C_D}(first mining round pending, or the launcher is older than 1.2.0: run '$0 update')${C_0}"
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
  if [[ -n "$err" ]]; then
    kv "last problem" "${C_Y}${err:0:90}${C_0}"
    if grep -qE 'not answering|cannot connect|connection refused|timeout' <<<"$err"; then
      note "                    the node did not answer. The launcher fails over to another node by itself; to"
      note "                    make it permanent: $0 set-node http://IP:8000   (or list backups in $WALLET_DIR/nodes.list)"
    fi
  fi
  if [[ -n "$failover" ]]; then kv "failover" "${C_Y}${failover:0:90}${C_0}"; fi
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
      local pwi="${pw%.*}" pli="${plim%.*}"      # nvidia-smi prints "450.00": shell arithmetic needs whole numbers
      if (( util >= 90 )) && [[ "$pwi" =~ ^[0-9]+$ && "$pli" =~ ^[0-9]+$ ]] && (( pli > 0 && pwi * 100 / pli < 55 )); then
        note "                    full load but only $((pwi * 100 / pli))% of the power limit: the cores are stalled, not computing (see: $0 bench)"
      fi
    done <<<"$smi"
  fi
  local log; log=$(miner_log)
  [[ -n "$log" ]] || return 0
  local rate="" samples="" ok_gpus="" bad="" sv=0 rj=0 refuse="" unreach="" restarts=0 fatal="" key val
  local gk="" gtot="" gmtot="" gl=() gid gname glast gavg gpeak gvar gbps
  while IFS=$'\t' read -r key val gname glast gavg gpeak gvar gbps; do
    case "$key" in rate) rate=$val;; samples) samples=$val;; ok) ok_gpus=$val;; bad) bad=$val;; sv) sv=$val;; rj) rj=$val;;
                   refuse) refuse=$val;; unreach) unreach=$val;; restarts) restarts=$val;; fatal) fatal=$val;;
                   k) gk=$val;; tot) gtot=$val;; mtot) gmtot=$val;;
                   gpu) gl+=("$val|$gname|$glast|$gavg|$gpeak|$gvar|$gbps");; esac
  done < <(awk '
    { if (match($0,/mining\.\.\. [0-9_]+ attempts +[0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,ff," "); r=ff[4]; gsub(/_/,"",r); mt=r }
      if (match($0,/attempts at [0-9_]+ H\/s/)) { s=substr($0,RSTART,RLENGTH); n=split(s,ff," "); r=ff[3]; gsub(/_/,"",r); if (r+0>0) mt=r }
      if (index($0,"totals:")) tot=substr($0,index($0,"totals:"))
      if (match($0,/difficulty K=[0-9]+/)) kk=substr($0,RSTART+13,RLENGTH-13) }
    index($0,"[seeker]") {
      if (match($0,/gpu[0-9]+ [^|]* rate [0-9]+ H\/s/)) {   # per card: "gpu0 <name> rate N H/s" (every 10 s)
        t=substr($0,RSTART,RLENGTH); id=t; sub(/ .*/,"",id); nm=t; sub(/^gpu[0-9]+ /,"",nm); sub(/ rate [0-9]+ H\/s$/,"",nm)
        r=t; sub(/ H\/s$/,"",r); sub(/.* /,"",r); gname[id]=nm; glast[id]=r; gsum[id]+=r; gcnt[id]++; if (r+0>gpeak[id]) gpeak[id]=r+0 }
      if (match($0,/gpu[0-9]+ .*(using|kernel variant) [A-Za-z0-9\/]+/)) { t=substr($0,RSTART,RLENGTH); id=t; sub(/ .*/,"",id); v=t; sub(/.*(using|kernel variant) /,"",v); gvar[id]=v }
      if (match($0,/gpu[0-9]+ .*: [0-9]+ blocks x [0-9]+ threads per launch \([0-9]+ per SM\)/)) { t=substr($0,RSTART,RLENGTH); id=t; sub(/ .*/,"",id); b=t; sub(/.*\(/,"",b); sub(/ per SM\).*/,"",b); gbps[id]=b }
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
          print "refuse\t" f; print "unreach\t" un+0; print "restarts\t" rs+0; print "fatal\t" fa
          print "k\t" kk; print "tot\t" tot; print "mtot\t" mt
          for (id in gname) printf "gpu\t%s\t%s\t%d\t%d\t%d\t%s\t%s\n", id, gname[id], glast[id], gsum[id]/gcnt[id], gpeak[id], gvar[id], gbps[id] }' <<<"$log") || true

  if [[ -z "$ok_gpus" && ${#gl[@]} -gt 0 ]]; then ok_gpus="${#gl[@]} card(s) (running: a card only reports a rate after the check)"; fi
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
  # --- per card: rate, kernel level, and what that card earns (best case, same formula as the MINER block)
  if (( ${#gl[@]} > 0 )); then
    mapfile -t gl < <(printf '%s\n' "${gl[@]}" | sort -V)
    local kk="$gk" minted="" re2='minted ([0-9_]+) \| K=([0-9]+)'
    if [[ "$gtot" =~ $re2 ]]; then minted=${BASH_REMATCH[1]//_/}; [[ -n "$kk" ]] || kk=${BASH_REMATCH[2]}; fi
    local sumr=0 e; for e in "${gl[@]}"; do IFS='|' read -r gid gname glast gavg gpeak gvar gbps <<<"$e"; sumr=$((sumr + gavg)); done
    # the rig as a whole: probability of a solution per slot with every card (a single key: one solution per slot at most)
    local rigday=0
    if [[ "$kk" =~ ^[0-9]+$ ]] && (( sumr > 0 )); then
      rigday=$(awk -v r="$sumr" -v k="$kk" -v s="${NET_SLOT_SEC:-10.24}" -v rw="${NET_REWARD:-95000000}" -v u="${NET_UNIT:-1000000}" \
        'BEGIN{ p=1-exp(-r*s/2^k); printf "%.3f", p*86400/s*rw/u }') || rigday=0
    fi
    printf '\n'
    kv "per card" "${C_D}average over the visible log · earnings = best case (no rivals), split by share of the rig's hashrate${C_0}"
    for e in "${gl[@]}"; do
      IFS='|' read -r gid gname glast gavg gpeak gvar gbps <<<"$e"
      local share="" day="" sofar=""
      if (( sumr > 0 )); then share=$(awk -v a="$gavg" -v t="$sumr" 'BEGIN{printf "%.0f", a*100/t}'); fi
      if [[ "$rigday" != 0 ]] && (( sumr > 0 )); then day=$(awk -v d="$rigday" -v a="$gavg" -v t="$sumr" 'BEGIN{printf "%.2f", d*a/t}'); fi
      if [[ -n "$minted" && "$minted" != 0 ]] && (( sumr > 0 )); then sofar=$(awk -v m="$minted" -v a="$gavg" -v t="$(( gmtot > sumr ? gmtot : sumr ))" -v u="${NET_UNIT:-1000000}" 'BEGIN{printf "%.4f", m*a/t/u}'); fi
      printf '  %s%-17s%s %s%-6s%s %s%-26s%s %s%11s%s %s(avg %s, peak %s)%s\n' "$C_D" "" "$C_0" "$C_C" "${gid}" "$C_0" "$C_W" "${gname:0:26}" "$C_0" "$C_B$C_G" "$(fmt_rate "$glast")" "$C_0" "$C_D" "$(fmt_rate "$gavg")" "$(fmt_rate "$gpeak")" "$C_0"
      printf '  %s%-17s%s %s%s%s%s%s%s%s\n' "$C_D" "" "$C_0" "$C_D" "kernel ${gvar:-?}${gbps:+ · $gbps blocks/SM}" "$C_0" \
        "${share:+   ${C_D}${share}% of GPU hashrate${C_0}}" "${day:+   ${C_G}≈ ${day} ${NET_TICKER:-PROX}/day${C_0}${C_D} (best case)${C_0}}" "${sofar:+   ${C_D}this session so far: ${C_G}${sofar} ${NET_TICKER:-PROX}${C_0}${C_D} (its share of everything minted)${C_0}}"
    done
    if [[ "$gmtot" =~ ^[0-9]+$ ]] && (( gmtot > sumr )); then
      if [[ "$POOL_ROLE" == server ]]; then kv "" "${C_D}CPU threads + pool workers: $(fmt_rate $((gmtot - sumr)))  (total ${C_W}$(fmt_rate "$gmtot")${C_0}${C_D} minus the cards of THIS machine — see POOL)${C_0}"
      else kv "" "${C_D}CPU threads: $(fmt_rate $((gmtot - sumr)))  (total ${C_W}$(fmt_rate "$gmtot")${C_0}${C_D} minus the cards)${C_0}"; fi
    fi
    note "                    all cards mine under ONE key, so there is no per-card payout: this splits the rig's income by hashrate"
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
  if [[ "$POOL_ROLE" == server ]]; then note "  pool server: $0 pool (members, join command) · $0 pool rotate (new token)"; fi
  if [[ -n "$OWN_NODE" ]]; then note "  own node: $0 logs node · $0 node-reset (wipe DB, new snapshot) · $0 install --no-own-node (use a remote node again)"; fi
  local me="${0##*/}" d="${C_D}·${C_0}"
  printf '\n  %s' "$C_D"; rule $((DASH_W - 4)); printf '%s\n' "$C_0"
  printf '  %s%s logs%s %s %s%s watch [secs]%s %s %s%s check%s %s %s%s gpu%s %s %s%s bench%s %s %s%s set-node URL%s %s %s%s key%s %s %s%s update%s\n' \
    "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0" "$d" "$C_C" "$me" "$C_0"
}

dash_software_worker() {
  sect "SOFTWARE"
  kv "installed" "gpu_seeker${MINER_WORKERS:+ + nonce_seeker (CPU)}   ${C_D}script $SCRIPT_VERSION · pool worker${C_0}"
  if [[ "${USE_GPU:-0}" == 1 ]] && gpu_stale 2>/dev/null; then kv "" "${C_Y}the seeker is older than this script or the proxima source — run: $0 update${C_0}"; fi
}
cmd_status() {
  load_install
  PUB_CUR=""; PUB_LRB=""
  if [[ "$POOL_ROLE" == worker ]]; then
    dash_banner; dash_pool_worker; dash_gpu; dash_software_worker
    printf '\n  %s' "$C_D"; rule $((DASH_W - 4)); printf '%s\n' "$C_0"
    printf '  %s%s logs%s · %s%s watch%s · %s%s pool%s · %s%s check%s · %s%s gpu%s · %s%s bench%s · %s%s update%s\n' "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0" "$C_C" "${0##*/}" "$C_0"
    return 0
  fi
  dash_banner
  dash_network
  dash_node
  dash_wallet
  dash_miner
  dash_pool_server
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
  # `logs node`: the local node's log (own-node mode)
  if [[ "${ARGS[0]:-}" == node ]]; then
    node_managed || die "no own node here (install with --own-node)"
    if node_unit_installed; then as_root journalctl -u "$NODE_SVC" -f -n 100 -o cat || true
    else tail -n 100 -F "$NODE_LOG" | tr '\r' '\n' || true; fi
    return 0
  fi
  # not 'exec': as_root is a shell function. Ctrl-C ends the follow; that is not an error.
  if unit_installed; then as_root journalctl -u "$MINER_SVC" -f -n 100 -o cat || true
  else tail -n 100 -F "$LOG_FILE" | tr '\r' '\n' || true; fi
}

# =============================================================================
#  check: can this VPS use your node?  (works before anything is installed)
# =============================================================================
cmd_check() {
  load_conf
  if [[ "$POOL_ROLE" == worker ]]; then cmd_check_worker; return 0; fi
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
    if seeker_installed; then
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
      seeker_installed || die "the GPU seeker is not installed — run: $0 install --gpu"
      detect_gpus; (( GPU_N > 0 )) || die "no NVIDIA GPU answers nvidia-smi"
      USE_GPU=1; WITH_SEEKER=1
      local rc=0; gpu_selftest --check || rc=$?; (( rc == 0 )) || die "GPU self-test failed (exit $rc) — not switching the GPU on"
      save_conf; write_launcher; write_unit; svc restart; ok "GPU mining is on";;
    off)
      load_install
      if [[ "$POOL_ROLE" == worker && ! -x "$CPU_SEEKER_BIN" ]]; then die "this worker has no CPU seeker installed: switching the GPU off would leave it with nothing to run. Re-install it CPU-only:  $0 install --pool-worker $POOL_URL --pool-token T --key-file F --no-gpu --workers N"; fi
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
    node          : $(if [[ -n "$OWN_NODE" ]]; then echo "${C_G}YOUR OWN, on this machine${C_0} — $NODE_URL ($(if node_unit_installed; then echo "systemd: $NODE_SVC"; else echo "background process, log $NODE_LOG"; fi)); the miner uses ONLY this node"; else echo "(remote) $NODE_URL"; fi)
    miner         : $(if [[ "${USE_GPU:-0}" == 1 ]]; then echo "GPU ($([[ -n "$GPU_DEVICES" ]] && echo "device(s) $GPU_DEVICES" || echo "all $GPU_N"), intensity $GPU_INTENSITY) + "; fi)$MINER_WORKERS CPU thread(s)$([[ "${MINER_MAX_KHS:-0}" != 0 ]] && echo ", cap $MINER_MAX_KHS KH/s") — $(if unit_installed; then echo "systemd: $MINER_SVC"; else echo "background process (pid file $PID_FILE)"; fi)
    wallet        : $WALLET_DIR   holder ${C_W}$holder${C_0}

  ${C_B}Everyday commands${C_0}   (shortcut: ${C_C}${SELF_SHORT:-$0}${C_0} <command>)
    status | watch   dashboard     logs   follow the miner     check   test node link + GPU
    gpu [on|off|test]   GPU setup / self-test     bench   measure the GPU hashrate
    set-node URL     use another node     key   change wallet key     update   rebuild from master
$([[ -n "$OWN_NODE" ]] && printf '    logs node        follow the local node   node-reset   wipe its database and re-sync from a snapshot\n')

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
  warn "the GPU self-test failed (exit $rc) — see the lines above. Possible causes: driver too old for this card (CUDA 12 needs >= 525, RTX 50xx >= 570), no libcuda, GPU busy/hung."
  warn "falling back to CPU-only mining; fix the cause, then run: $0 gpu on"
  USE_GPU=0; WITH_SEEKER=0
  if [[ ! "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then MINER_WORKERS=$(nproc 2>/dev/null || echo 1); fi
  return 1
}

# --own-node: ask once (a saved choice is kept); a node on THIS machine, the miner talks only to it
node_api_port_default() { if node_managed; then node_api_port; else echo 8000; fi; }
choose_own_node() {
  if [[ -z "$OWN_NODE" && "$OWN_NODE_ASKED" != 1 && -z "$NODE_URL" ]] && (( HAVE_TTY )); then
    step "Node"
    log "Option A: use a node somewhere else (default: $DEFAULT_NODE_URL)"
    log "Option B: run your OWN node on this machine and mine ONLY through it (no other node is contacted by the miner)"
    if confirm "Run your own node here and mine only through it?" n; then OWN_NODE=1; fi
  fi
  OWN_NODE_ASKED=1
  norm_flags
  [[ -n "$OWN_NODE" ]] || return 0
  step "Own node + CPU + GPU miner"
  ok "the node runs here ($NODE_DIR); the miner uses http://127.0.0.1:$(node_api_port_default) and nothing else"
  warn "a first start downloads the ledger snapshot and syncs: a few minutes; mining starts when the node is synced"
  warn "the node still needs its peers (UDP $(node_peer_port_default)) to receive and publish transactions: that is the network, not the miner"
  local gb; gb=$(df -Pk "$MINER_HOME" | awk 'NR==2{print int($4/1024/1024)}')
  (( gb >= 8 )) || warn "only ${gb} GB free in $MINER_HOME: the node database grows over time"
  return 0
}
node_peer_port_default() { if node_managed; then node_peer_port; else echo 4000; fi; }
OWN_NODE_ASKED=0
# switching back from own-node mode to a remote node: the local node is stopped and its unit removed (database kept)
retire_own_node() {
  node_managed || return 0
  [[ -n "$(node_pid)" || -f "$NODE_UNIT" ]] || return 0
  log "own-node mode is off: stopping the local node (its database in $NODE_DIR is kept)"
  node_svc stop 2>/dev/null || true
  if have_systemd && [[ -f "$NODE_UNIT" ]]; then as_root systemctl disable "$NODE_SVC" 2>/dev/null || true; as_root rm -f "$NODE_UNIT"; as_root systemctl daemon-reload; fi
  return 0
}

cmd_install() {
  step "Proxima CPU+GPU miner $SCRIPT_VERSION — user=$RUN_USER home=$MINER_HOME branch=$BRANCH"
  local forced_gpu="$USE_GPU"
  load_conf
  choose_pool_role
  if [[ "$POOL_ROLE" == worker ]]; then cmd_install_worker "$forced_gpu"; return 0; fi
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
  choose_own_node
  if [[ -n "$OWN_NODE" ]]; then NODE_URL="http://127.0.0.1:$(node_api_port_default)"; setup_ledger_apis
  else
    choose_node
    setup_ledger_apis
  fi
  choose_key
  if [[ "$POOL_ROLE" == server ]]; then choose_pool_server; fi
  choose_gpu
  if [[ "$USE_GPU" != 1 && ! "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]] && [[ "$POOL_ROLE" != server || ! "$MINER_WORKERS" =~ ^[0-9]+$ ]]; then MINER_WORKERS=""; fi
  choose_workers
  check_clock
  fetch_source; check_compat
  ensure_go
  build_proxi
  if [[ -n "$OWN_NODE" ]]; then build_node_binary; fi
  if [[ "$USE_GPU" == 1 ]]; then ensure_cuda; ensure_rust; build_gpu_seeker; fi
  retire_cpu_service
  svc stop 2>/dev/null || true
  if [[ -n "$OWN_NODE" ]]; then node_svc stop 2>/dev/null || true; fi
  install_binary
  if [[ -n "$OWN_NODE" ]]; then install_node_binary; fi
  if [[ "$USE_GPU" == 1 ]]; then
    install_gpu_files
    if ! gpu_self_check_or_fallback; then ok "continuing as a CPU miner with $MINER_WORKERS thread(s)"; fi
  fi
  if [[ -n "$OWN_NODE" ]]; then configure_own_node; write_node_unit; fi
  apply_key
  save_conf
  write_launcher
  write_unit
  if [[ "$POOL_ROLE" == server ]]; then pool_firewall; fi
  if [[ -n "$OWN_NODE" ]]; then node_svc enable; else retire_own_node; fi
  svc enable
  pick_self_short
  if [[ -n "$OWN_NODE" ]]; then wait_own_node_sync; fi
  summary
  if [[ "$POOL_ROLE" == server ]]; then summary_pool_server; fi
  sleep 4
  if [[ -t 1 ]]; then cmd_status || true; fi
}

cmd_startstop() { # with an own node: the node starts first and stops last
  local a="$1"
  if [[ -z "$OWN_NODE" ]]; then svc "$a"; return 0; fi
  case "$a" in
    start)   node_svc start; svc start;;
    stop)    svc stop; node_svc stop;;
    restart) svc stop; node_svc restart; svc start;;
  esac
}

cmd_set_node() {
  load_install
  [[ "$POOL_ROLE" != worker ]] || die "a pool worker has no node: it talks to the pool server only. Change the server with:  $0 install --pool-worker URL --pool-token T --key-file F"
  local url="${ARGS[0]:-$NODE_URL_ARG}"
  [[ -n "$url" ]] || die "usage: $0 set-node http://IP:8000"
  if [[ -n "$OWN_NODE" ]]; then die "own-node mode is on: the miner uses only your local node. To use a remote node instead:  $0 install --no-own-node --node-url $url"; fi
  NODE_URL="$url"; choose_node
  profile_apply "$NODE_URL"
  setup_ledger_apis
  svc restart
  ok "miner now uses $NODE_URL"
}

cmd_key() {
  load_install
  [[ "$POOL_ROLE" != worker ]] || die "a pool worker must use the pool server's key: re-install it with  $0 install --pool-worker $POOL_URL --pool-token T --key-file /path/to/proxima.key"
  [[ -x "$BIN_DIR/proxi" ]] || die "proxi not found in $BIN_DIR — run: $0 install"
  choose_key
  [[ "$KEY_MODE" == keep ]] && { ok "key unchanged"; return 0; }
  svc stop 2>/dev/null || true
  apply_key
  write_launcher
  svc start
  if [[ "$POOL_ROLE" == server ]]; then warn "the pool workers still hold the OLD key and are refused now: copy the new $WALLET_DIR/proxima.key to each of them and re-run  $(pool_join_command)"; fi
}

launcher_stale() { [[ -f "$LAUNCHER" ]] && ! grep -q "generated by proxima-gpu-miner.sh $SCRIPT_VERSION " "$LAUNCHER"; }

cmd_update() {
  load_conf
  if [[ "$POOL_ROLE" == worker ]]; then load_install; cmd_update_worker; return 0; fi
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
    retire_cpu_service; install_self; save_conf; write_launcher; write_unit; if [[ -n "$OWN_NODE" ]]; then write_node_unit; fi; svc restart; ok "launcher + script updated, miner restarted"; return 0
  fi
  if (( ! need_proxi && ! need_gpu )); then
    ok "already running $old_v ($(printf '%.10s' "$have"))$([[ "$USE_GPU" == 1 ]] && echo ", GPU seeker up to date") — nothing to do"; return 0
  fi
  if (( need_proxi )); then ensure_go; build_proxi; log "upgrade: ${old_v:-none} -> $SRC_VERSION"; if [[ -n "$OWN_NODE" ]]; then build_node_binary; fi; fi
  if (( need_gpu )); then ensure_cuda; ensure_rust; build_gpu_seeker; fi
  retire_cpu_service
  svc stop 2>/dev/null || true
  if [[ -n "$OWN_NODE" ]] && (( need_proxi )); then node_svc stop 2>/dev/null || true; fi
  if (( need_proxi )); then install_binary; if [[ -n "$OWN_NODE" ]]; then install_node_binary; fi; fi
  if (( need_gpu )); then install_gpu_files; gpu_self_check_or_fallback || true; fi
  migrate_wallet_profile
  if [[ -n "$OWN_NODE" ]]; then
    refresh_public_peers; write_node_unit
    if own_node_on_other_ledger; then warn "your node is on an OLD ledger (the network was reset): run  $0 node-reset"; fi
    node_svc start
  fi
  save_conf; write_launcher; write_unit
  svc start
  ok "updated to $SRC_VERSION"
}

cmd_uninstall() {
  load_install 2>/dev/null || load_conf
  confirm "Stop and remove the miner service and the proxi / gpu_seeker binaries? (the wallet in $WALLET_DIR is kept)" y || exit 0
  svc stop 2>/dev/null || true
  node_svc stop 2>/dev/null || true
  if have_systemd && [[ -f "$NODE_UNIT" ]]; then as_root systemctl disable "$NODE_SVC" 2>/dev/null || true; as_root rm -f "$NODE_UNIT"; as_root systemctl daemon-reload; fi
  if have_systemd && [[ -f "$UNIT_FILE" ]]; then
    as_root systemctl disable "$MINER_SVC" 2>/dev/null || true
    as_root rm -f "$UNIT_FILE"; as_root systemctl daemon-reload
  fi
  as_root rm -f "$BIN_DIR/proxi" "$BIN_DIR/proxi.prev" "$BIN_DIR/proxima" "$BIN_DIR/proxima.prev" "$SEEKER_BIN" "$CPU_SEEKER_BIN" "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin"
  if [[ -L /usr/local/bin/proxima-gpu-miner ]]; then as_root rm -f /usr/local/bin/proxima-gpu-miner; fi
  rm -rf "$SEEKER_PATH_DIR" "$GPU_INFO" "$GPU_SRC" "$GPU_TARGET" "$GPU_BUILD"
  rm -f "$LAUNCHER" "$PID_FILE" "$CPU_SEEKER_STAMP"
  if [[ "$POOL_ROLE" == server ]]; then log "the firewall rule for the pool port ${POOL_PORT:-$(pool_server_port)} (if any) is left in place: remove it yourself (ufw delete allow ${POOL_PORT:-$(pool_server_port)}/tcp)"; fi
  ok "removed. Your key and data remain in $MINER_HOME"
  if [[ -d "$NODE_DIR" ]]; then log "the node configuration + database stay in $NODE_DIR (host key, snapshot)"; fi
  log "the private toolchains stay in $MINER_HOME/.toolchain (Go, Rust, CUDA compiler: ~1.5 GB) — they are reused by a later install"
  log "to delete everything (the Go module cache is read-only):  chmod -R u+w $MINER_HOME && rm -rf $MINER_HOME   ${C_Y}— back up proxima.key first${C_0}"
}

usage() {
  if [[ -n "$SELF_PATH" ]]; then awk 'NR>1 && /^#/{print; next} NR>1{exit}' "$SELF_PATH" | sed 's/^# \{0,1\}//' | grep -v '^====='
  else # piped (curl | bash): the file is not there to re-read
    cat <<EOT
proxima-gpu-miner.sh $SCRIPT_VERSION — standalone CPU + NVIDIA GPU (RTX 20xx / 30xx / 40xx / 50xx) Proxima miner, mines against a remote node

  install | status | watch [secs] | check | gpu [on|off|test] | bench [secs] | logs | start | stop | restart
  set-node URL | node-reset | key | update | uninstall | help

  pool     : pool [show|join|rotate]   several VPS = ONE miner:  install --pool-server   |   install --pool-worker http://SERVER:8100 --pool-token T --key-file F
  options: --node-url URL  --own-node (run your own node here, mine only through it)  --key-file FILE  --generate-key
           --workers N (0 = GPU only)  --max-khs X  --no-public
           --gpu | --no-gpu  --gpu-devices 0,1  --gpu-intensity low|normal|high  --gpu-arch "86 89"|all
           --home DIR  --branch NAME  --bin-dir DIR  --no-services  --no-color  -y
  Save the script to a file (or run 'install' once) to see the full manual: proxima-gpu-miner.sh help
EOT
  fi
}

# =============================================================================
#  POOL: several VPS act as ONE miner.
#    server  = the machine that runs `proxi node mine` (the only one that talks to the node and submits).
#              It opens mine.seeker.listen/token, so other machines can dial in and search for it.
#    worker  = a machine that only SEARCHES: gpu_seeker (GPU) and/or the reference nonce_seeker (CPU) dial the
#              server with the same wallet key.  No node, no proxi, no database on a worker.
#  The hashrates add up (every member searches its own random nonce range), the server submits once per
#  round, and the reward goes to the one key.  This is NOT a payout-sharing pool (see the notes printed by install).
# =============================================================================
POOL_DEFAULT_PORT=8100
pool_new_token() { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
pool_is_server() { [[ "$POOL_ROLE" == server ]]; }
pool_is_worker() { [[ "$POOL_ROLE" == worker ]]; }

# value of mine.seeker.<key> in the wallet profile ("" when unset)
profile_pool_get() { # profile_pool_get listen|token
  awk -v want="$1" '
    function ind(s){ match(s,/^[ \t]*/); return RLENGTH }
    /^mine:/ {inmine=1; next}
    inmine && /^[^ \t#]/ {inmine=0}
    inmine && /^[ \t]+seeker:/ {inseek=1; si=ind($0); next}
    inseek && /^[ \t]*[^ \t#]/ && ind($0)<=si {inseek=0}
    inseek && $0 ~ "^[ \t]+" want ":" { v=$0; sub(/^[^:]*:[ \t]*/,"",v); gsub(/["\047]/,"",v); sub(/[ \t]+#.*$/,"",v); print v; exit }
  ' "$WALLET_DIR/proxi.yaml" 2>/dev/null || true
}
# set (or, with empty values, clear) mine.seeker.listen / mine.seeker.token in the wallet profile
profile_pool_set() { # profile_pool_set <listen> <token>
  local y="$WALLET_DIR/proxi.yaml"
  [[ -f "$y" ]] || return 0
  awk -v L="$1" -v T="$2" '
    function ind(s){ match(s,/^[ \t]*/); return RLENGTH }
    function pad(n,  s,i){ s=""; for(i=0;i<n;i++) s=s " "; return s }
    function emit(n){ print pad(n) "listen: \"" L "\""; print pad(n) "token: \"" T "\"" }
    /^mine:/ { inmine=1; hasmine=1; print; next }
    inmine && /^[^ \t#]/ {
      if (!hasseek) { print "    seeker:"; emit(8); hasseek=1 }
      inmine=0; inseek=0
    }
    inmine && /^[ \t]+seeker:/ { print; hasseek=1; inseek=1; si=ind($0); emit(si+4); next }
    inseek {
      if ($0 ~ /^[ \t]*[^ \t#]/ && ind($0)<=si) inseek=0
      else if ($0 ~ /^[ \t]+(listen|token):/) next
    }
    { print }
    END {
      if (!hasmine) { print "mine:"; print "    seeker:"; emit(8) }
      else if (inmine && !hasseek) { print "    seeker:"; emit(8) }
    }
  ' "$y" | as_user tee "$y.new" >/dev/null
  as_user mv -f "$y.new" "$y"; as_user chmod 600 "$y"
}
pool_apply_profile() { # called after the wallet profile was (re)written: put the pool server settings in, or take them out
  [[ -f "$WALLET_DIR/proxi.yaml" ]] || return 0
  if pool_is_server; then
    [[ -n "$POOL_TOKEN" ]] || die "pool server without a token (internal)"
    profile_pool_set "${POOL_BIND:-0.0.0.0}:${POOL_PORT:-$POOL_DEFAULT_PORT}" "$POOL_TOKEN"
    ok "pool server: the miner serves search jobs on ${POOL_BIND:-0.0.0.0}:${POOL_PORT:-$POOL_DEFAULT_PORT} (token set in proxi.yaml)"
  elif [[ -n "$(profile_pool_get listen)" ]]; then
    profile_pool_set "" ""
    ok "pool server settings removed from the wallet profile"
  fi
  return 0
}

pool_firewall() { # server: let the workers in
  local port="${POOL_PORT:-$POOL_DEFAULT_PORT}" ip
  if [[ "${POOL_BIND:-0.0.0.0}" == 127.* ]]; then
    log "the job server listens on loopback only: workers must reach it through an SSH / WireGuard tunnel (see 'pool')"; return 0
  fi
  if command -v ufw >/dev/null 2>&1 && as_root ufw status 2>/dev/null | grep -q 'Status: active'; then
    if [[ -n "$POOL_ALLOW" ]]; then
      for ip in ${POOL_ALLOW//,/ }; do
        [[ "$ip" =~ ^[0-9a-fA-F:./]+$ ]] || { warn "not an IP address, skipped: $ip"; continue; }
        as_root ufw allow from "$ip" to any port "$port" proto tcp >/dev/null && ok "ufw: TCP $port open for $ip only"
      done
    else
      as_root ufw allow "$port/tcp" >/dev/null && ok "ufw: TCP $port open"
      warn "port $port is now open to the whole internet and protected by the token only — better: --pool-allow IP1,IP2 (your workers)"
    fi
  else
    if [[ -n "$POOL_ALLOW" ]]; then warn "no active ufw here: --pool-allow was NOT enforced. Restrict TCP $port to $POOL_ALLOW in your provider's firewall panel."
    else log "no active ufw: open TCP $port for your worker VPS in the provider's firewall panel (and restrict it to their IPs)"; fi
  fi
  return 0
}

# ---- asking -----------------------------------------------------------------
choose_pool_role() {
  [[ "$POOL_OFF" == 1 ]] && POOL_ROLE=""
  if [[ -z "$POOL_ROLE" && "$POOL_OFF" != 1 && "$POOL_ASKED" != 1 ]] && (( HAVE_TTY )); then
    step "Pool — combine several VPS into ONE miner?"
    cat <<EOT
  A pool here means: one machine (the ${C_B}server${C_0}) runs the miner, the other VPS (the ${C_B}workers${C_0}) only search nonces for it
  with the SAME wallet key. All hashrates add up, the server submits once, the reward goes to your one key.
    1) no - this machine mines on its own  ${C_D}(default)${C_0}
    2) this machine is the pool ${C_B}SERVER${C_0}  (it also mines itself, and gets a join command for the workers)
    3) this machine is a pool ${C_B}WORKER${C_0}   (GPU / CPU only, no node, no database; needs the server's URL + token + key)
EOT
    local c; ask c "choose" "1"
    case "$c" in 2|server) POOL_ROLE=server;; 3|worker) POOL_ROLE=worker;; *) POOL_ROLE="";; esac
  fi
  POOL_ASKED=1
  case "$POOL_ROLE" in ""|server|worker) ;; *) die "invalid pool role '$POOL_ROLE' (server | worker)";; esac
  return 0
}
POOL_ASKED=0

choose_pool_server() {
  step "Pool server"
  if [[ -z "$POOL_PORT" ]]; then ask POOL_PORT "port of the job server (the workers connect to it)" "$POOL_DEFAULT_PORT"; fi
  [[ "$POOL_PORT" =~ ^[0-9]+$ && "$POOL_PORT" -ge 1 && "$POOL_PORT" -le 65535 ]] || die "invalid pool port '$POOL_PORT'"
  POOL_BIND="${POOL_BIND:-0.0.0.0}"
  if [[ -z "$POOL_TOKEN" ]]; then POOL_TOKEN=$(pool_new_token); ok "generated the pool token"
  else [[ "$POOL_TOKEN" =~ ^[A-Za-z0-9._~-]{8,}$ ]] || die "the pool token must be 8+ characters of A-Z a-z 0-9 . _ ~ -"; fi
  ok "job server: $POOL_BIND:$POOL_PORT, token-protected"
  cat <<EOT
  ${C_Y}Read this once:${C_0}
    * every worker holds your wallet KEY in its memory while it searches (it needs it to compute the VRF).
      Use only machines you control, or a dedicated key whose rewards you sweep away from time to time.
    * the token travels over plain HTTP. Allow only your workers' IPs (--pool-allow) or use an SSH/WireGuard tunnel.
    * this is ONE miner with more hashrate, not a payout-sharing pool: all rewards go to the one key.
EOT
}

choose_pool_worker() {
  step "Pool worker — which pool server?"
  local try url
  for try in 1 2 3; do
    if [[ -z "$POOL_URL" || $try -gt 1 ]]; then ask url "URL of the pool server (like http://1.2.3.4:$POOL_DEFAULT_PORT)" "$POOL_URL"; else url="$POOL_URL"; fi
    [[ -n "$url" ]] || { warn "an URL is needed"; continue; }
    url="${url//[[:space:]]/}"; url="${url%/}"
    [[ "$url" =~ ^https?:// ]] || url="http://$url"
    [[ "${url#*//}" =~ :[0-9]+$ ]] || url="$url:$POOL_DEFAULT_PORT"
    POOL_URL="$url"
    if [[ -z "$POOL_TOKEN" ]]; then ask_secret POOL_TOKEN "pool token (from the server's install output / '$0 pool')"; fi
    [[ -n "$POOL_TOKEN" ]] || warn "no token given: only works if the server runs without one (loopback)"
    log "checking $POOL_URL …"
    pool_probe "$POOL_URL" "$POOL_TOKEN"
    case "$PP_CODE" in
      200) ok "the pool server answers in ${PP_RTT} ms and accepts the token"; break;;
      401) warn "the server is there but REFUSES the token"; POOL_TOKEN="";;
      0)   warn "no answer from $POOL_URL — is the miner on the server running, and does its firewall let this VPS ($(public_ip || echo '?')) in?";;
      *)   warn "unexpected answer HTTP $PP_CODE from $POOL_URL";;
    esac
    if (( ! HAVE_TTY )); then
      [[ "$PP_CODE" != 401 ]] || die "the pool server refused the token"
      warn "continuing anyway: the worker keeps retrying until the server is reachable"; break
    fi
    if (( try == 3 )); then confirm "install anyway (the worker keeps retrying)?" n || die "no working pool server given"; fi
  done
  [[ -n "$POOL_URL" ]] || die "no pool server URL given"
  POOL_NAME="${POOL_NAME:-$(hostname -s 2>/dev/null || echo worker)}"
  POOL_NAME="${POOL_NAME//[^A-Za-z0-9._-]/-}"
  ok "this worker is called '$POOL_NAME' on the server's dashboard"
}

# GET /seeker/job on a pool server: PP_CODE (0 = unreachable), PP_RTT (ms), PP_JOB (json)
PP_CODE=0; PP_RTT=""; PP_JOB=""
pool_probe() { # pool_probe URL TOKEN
  local url="$1" tok="${2:-}" out hdr=()
  PP_CODE=0; PP_RTT=""; PP_JOB=""
  [[ -z "$tok" ]] || hdr=(-H "Authorization: Bearer $tok")
  out=$(curl -s -m 8 -w '\n%{http_code} %{time_total}' ${hdr[@]+"${hdr[@]}"} "$url/seeker/job" 2>/dev/null) || { PP_CODE=0; return 0; }
  local last; last=$(tail -n1 <<<"$out")
  PP_CODE=${last%% *}; PP_RTT=$(awk -v t="${last##* }" 'BEGIN{printf "%d", t * 1000}')
  PP_JOB=$(sed '$d' <<<"$out")
  [[ "$PP_CODE" =~ ^[0-9]+$ ]] || PP_CODE=0
  return 0
}

# ---- worker: the reference CPU seeker (Rust, from the proxima source tree) -------------------------------------
CPU_TARGET_DIR_NAME="build/cpu-target"
cpu_seeker_stale() {
  [[ -x "$CPU_SEEKER_BIN" ]] || return 0
  [[ "$(cat "$CPU_SEEKER_STAMP" 2>/dev/null || true)" == "$(as_user git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || echo ?)" ]] && return 1
  return 0
}
build_cpu_seeker() {
  step "CPU seeker: the reference nonce_seeker (Rust)"
  local d="$SRC_DIR/proxi/node_cmd/mine/nonce_seeker" jobs=()
  [[ -f "$d/Cargo.toml" ]] || die "this proxima version has no nonce_seeker ($d) — pin the version the pool server runs with --branch <tag|commit>"
  if low_mem; then jobs=(-j 1); fi
  log "building nonce_seeker … (the first build downloads crates — a few minutes)"
  ( cd "$d" && rust_env env CARGO_TARGET_DIR="$MINER_HOME/$CPU_TARGET_DIR_NAME" RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet ${jobs[@]+"${jobs[@]}"} ) \
    || die "building the CPU seeker failed (scroll up for the compiler message)"
  as_user mkdir -p "$BUILD_OUT"
  as_user cp -f "$MINER_HOME/$CPU_TARGET_DIR_NAME/release/nonce_seeker" "$BUILD_OUT/pool_cpu_seeker.new"
  ok "nonce_seeker built"
}
install_cpu_seeker() {
  as_user mkdir -p "$BIN_DIR" 2>/dev/null || as_root mkdir -p "$BIN_DIR"
  put_file "$BUILD_OUT/pool_cpu_seeker.new" "$CPU_SEEKER_BIN" 0755
  as_user rm -f "$BUILD_OUT/pool_cpu_seeker.new"
  as_user git -C "$SRC_DIR" rev-parse HEAD | as_user tee "$CPU_SEEKER_STAMP" >/dev/null
  ok "CPU seeker -> $CPU_SEEKER_BIN"
}

# ---- worker: the key (import only: it has to be the pool's key) -------------------------------------------------
choose_worker_key() {
  step "Wallet key of the pool (the SAME proxima.key as on the pool server)"
  local dst="$WALLET_DIR/proxima.key"
  if [[ -z "$KEY_FILE" ]]; then
    if (( HAVE_TTY )); then
      local def=""; [[ -f "$dst" ]] && def="$dst"
      cat <<EOT
  A worker computes the VRF with your wallet key, so it needs the key file of the pool server
  (on the server: ${C_W}$WALLET_DIR/proxima.key${C_0}). Copy it here first, e.g. from the server:
      scp $RUN_USER@SERVER:$WALLET_DIR/proxima.key ~/proxima.key
EOT
      local p; ask p "path to proxima.key  (empty = paste the JSON here)" "$def"
      if [[ -n "$p" ]]; then KEY_FILE=$(expand_path "$p"); [[ -f "$KEY_FILE" ]] || die "file not found: $KEY_FILE"
      else paste_key; fi
    elif [[ -f "$dst" ]]; then KEY_FILE="$dst"
    else die "a pool worker needs the pool's key: --key-file /path/to/proxima.key"; fi
  else KEY_FILE=$(expand_path "$KEY_FILE"); [[ -f "$KEY_FILE" ]] || die "key file not found: $KEY_FILE"; fi
  key_json_ok "$KEY_FILE" || die "$KEY_FILE does not look like a Proxima key file (the JSON written by proxi)"
  ok "key: $(key_encrypted "$KEY_FILE" && echo 'encrypted' || echo 'plain') keystore, holder $(key_holder_of "$KEY_FILE" | cut -c1-16)…"
  if key_encrypted "$KEY_FILE"; then
    local p="${PROXIMA_KEY_PASSPHRASE:-}"
    if [[ -z "$p" ]]; then (( HAVE_TTY )) || die "the key is encrypted: set PROXIMA_KEY_PASSPHRASE (no terminal to ask)"; ask_secret p "passphrase of the key"; fi
    PROXIMA_KEY_PASSPHRASE="$p"; export PROXIMA_KEY_PASSPHRASE
    printf 'PROXIMA_KEY_PASSPHRASE=%q\n' "$p" | as_user tee "$ENV_FILE" >/dev/null; as_user chmod 600 "$ENV_FILE"
    ok "passphrase stored in $ENV_FILE (mode 600); it is checked when the GPU self-test loads the key"
  else as_user rm -f "$ENV_FILE"; fi
  as_user mkdir -p "$WALLET_DIR"; as_user chmod 700 "$WALLET_DIR"
  if [[ "$KEY_FILE" != "$dst" ]]; then
    [[ ! -f "$dst" ]] || { as_user mkdir -p "$BACKUP_DIR"; as_user cp -f "$dst" "$BACKUP_DIR/proxima.key.$(date +%Y%m%d-%H%M%S)"; }
    as_user install -m 600 "$KEY_FILE" "$dst"
  fi
  as_user chmod 600 "$dst"
  KEY_MODE=import
}
# load the key once with the installed GPU seeker: a wrong passphrase / damaged file shows up now, not as a restart loop
worker_key_check() {
  [[ "${USE_GPU:-0}" == 1 && -x "$SEEKER_BIN" ]] || return 0
  seeker_env
  local out rc=0
  out=$(as_user env "${SEEKER_ENV[@]}" PROXIMA_KEY_PASSPHRASE="${PROXIMA_KEY_PASSPHRASE:-}" timeout 300 "$SEEKER_BIN" --check --key-file "$WALLET_DIR/proxima.key" 2>&1 </dev/null) || rc=$?
  if (( rc == 0 )); then ok "the key loads and every GPU reproduces the CPU reference with it"; return 0; fi
  warn "loading the key / the GPU check failed:"; while IFS= read -r l; do [[ -n "$l" ]] && printf '    %s\n' "$l"; done <<<"$out"
  return 1
}
# does the server mine under this worker's key?  (only knowable while the server has a job)
POOL_KEYMATCH=""   # yes | no | ""
pool_key_match() { # after pool_probe
  POOL_KEYMATCH=""
  local mine theirs; mine=$(jq -r '.public_key // empty' "$WALLET_DIR/proxima.key" 2>/dev/null || true)
  theirs=$(jq -r '.pubkey // empty' <<<"$PP_JOB" 2>/dev/null || true)
  [[ -n "$mine" && -n "$theirs" ]] || return 0
  if [[ "${mine,,}" == "${theirs,,}" ]]; then POOL_KEYMATCH=yes; else POOL_KEYMATCH=no; fi
}

# ---- worker: launcher ----------------------------------------------------------------------------------------------
write_launcher_worker() {
  local body name="${POOL_NAME:-worker}"
  body=$(
    printf '#!/usr/bin/env bash\n# generated by proxima-gpu-miner.sh %s — pool worker (searches for %s); regenerated on install/update/gpu changes\n' "$SCRIPT_VERSION" "$POOL_URL"
    printf 'WALLET_DIR=%q\nENV_FILE=%q\nPOOL_URL=%q\nPOOL_TOKEN=%q\nNAME=%q\nUSE_GPU=%q\nCPU_THREADS=%q\nGPU_BIN=%q\nCPU_BIN=%q\nKEY=%q\n' \
      "$WALLET_DIR" "$ENV_FILE" "$POOL_URL" "$POOL_TOKEN" "$name" "${USE_GPU:-0}" "${MINER_WORKERS:-0}" "$SEEKER_BIN" "$CPU_SEEKER_BIN" "$WALLET_DIR/proxima.key"
    if [[ "${USE_GPU:-0}" == 1 ]]; then
      seeker_env
      local e; for e in "${SEEKER_ENV[@]}"; do printf 'export %q\n' "$e"; done
    fi
    cat <<'EOT'
cd "$WALLET_DIR" || exit 1
if [ -r "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
tok=(); [ -n "$POOL_TOKEN" ] && tok=(--token "$POOL_TOKEN")

# Both seekers dial the pool server themselves and retry (with back-off) while it is down or restarting.
# The sed prefix lets `status` tell the GPU lines from the CPU lines.
pids=()
if [ "$USE_GPU" = 1 ]; then
  echo "GPU seeker '$NAME-gpu' -> $POOL_URL"
  "$GPU_BIN" --proxi "$POOL_URL" "${tok[@]}" --key-file "$KEY" --name "$NAME-gpu" > >(sed -u 's/^/[seeker] /') 2>&1 &
  pids+=($!)
fi
if [ "$CPU_THREADS" -ge 1 ] 2>/dev/null; then
  echo "CPU seeker '$NAME-cpu' ($CPU_THREADS thread(s)) -> $POOL_URL"
  "$CPU_BIN" --proxi "$POOL_URL" "${tok[@]}" --key-file "$KEY" --threads "$CPU_THREADS" --name "$NAME-cpu" > >(sed -u 's/^/[cpu-seeker] /') 2>&1 &
  pids+=($!)
fi
[ "${#pids[@]}" -gt 0 ] || { echo "nothing to run: neither a GPU nor CPU threads are enabled"; sleep 30; exit 1; }

stop_all() { kill -TERM "${pids[@]}" 2>/dev/null; for _ in 1 2 3 4 5 6 7 8 9 10; do
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [ "$alive" = 0 ] && break; sleep 1; done
  kill -KILL "${pids[@]}" 2>/dev/null; wait 2>/dev/null; }
trap 'stop_all; exit 0' TERM INT
while :; do
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "a seeker exited (pid $p) - restarting the worker in 15 s"; stop_all; sleep 15 & wait $!; exec "$0" "$@"; }; done
  sleep 5 & wait $! || true
done
EOT
  )
  printf '%s\n' "$body" | as_user tee "$LAUNCHER" >/dev/null
  as_user chmod 700 "$LAUNCHER"
  ok "launcher: $LAUNCHER (pool worker)"
}

# ---- worker: install / update --------------------------------------------------------------------------------------
cmd_install_worker() { # cmd_install_worker <the --gpu/--no-gpu given on the command line, or "">
  local forced_gpu="${1:-}"
  warn "this machine becomes a pool WORKER: it only searches for the pool server — no node, no wallet database, no proxi binary"
  if [[ -n "$OWN_NODE" ]]; then OWN_NODE=""; retire_own_node; fi     # a worker runs no node (an earlier own-node install is retired, its database kept)
  SAVED_HAD_GPU=0
  if [[ -z "$forced_gpu" ]]; then if [[ "$USE_GPU" == 1 ]]; then SAVED_HAD_GPU=1; else USE_GPU=""; fi; fi
  detect_gpus
  WITH_SEEKER=1                      # Rust + a C compiler are needed for the CPU seeker as well
  preflight
  ensure_deps
  SAVED_WORKERS="$MINER_WORKERS"
  choose_pool_worker
  choose_worker_key
  choose_gpu
  if [[ "$USE_GPU" != 1 && ! "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then MINER_WORKERS=""; fi
  WITH_SEEKER=1
  choose_workers
  check_clock
  fetch_source
  if [[ "$USE_GPU" == 1 ]]; then ensure_cuda; ensure_rust; build_gpu_seeker; else ensure_rust; fi
  svc stop 2>/dev/null || true
  if [[ "$USE_GPU" == 1 ]]; then
    install_gpu_files
    if ! gpu_self_check_or_fallback; then ok "continuing as a CPU-only worker with $MINER_WORKERS thread(s)"; fi
    worker_key_check || { USE_GPU=0; [[ "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]] || MINER_WORKERS=$(nproc 2>/dev/null || echo 1); warn "the GPU part is off until this is fixed (then: $0 gpu on)"; }
  fi
  if [[ "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then build_cpu_seeker; install_cpu_seeker; fi
  install_self
  save_conf
  write_launcher_worker
  write_unit
  svc enable
  pick_self_short
  sleep 3
  summary_worker
  if [[ -t 1 ]]; then sleep 2; cmd_status || true; fi
}

cmd_update_worker() {
  step "Updating the pool worker — branch $BRANCH"
  [[ "$USE_GPU" == 1 ]] && WITH_SEEKER=1
  WITH_SEEKER=1
  preflight; ensure_deps
  fetch_source
  local need_gpu=0 need_cpu=0
  if [[ "$USE_GPU" == 1 ]] && { gpu_stale || [[ -n "${FORCE_REBUILD:-}" ]]; }; then need_gpu=1; fi
  if [[ "$MINER_WORKERS" =~ ^[1-9][0-9]*$ ]] && { cpu_seeker_stale || [[ -n "${FORCE_REBUILD:-}" ]]; }; then need_cpu=1; fi
  if (( ! need_gpu && ! need_cpu )); then
    if launcher_stale; then install_self; save_conf; write_launcher_worker; write_unit; svc restart; ok "launcher + script updated, worker restarted"; return 0; fi
    ok "the worker is up to date — nothing to do"; return 0
  fi
  if (( need_gpu )); then ensure_cuda; ensure_rust; build_gpu_seeker; fi
  if (( need_cpu )); then ensure_rust; build_cpu_seeker; fi
  svc stop 2>/dev/null || true
  if (( need_gpu )); then install_gpu_files; gpu_self_check_or_fallback || true; fi
  if (( need_cpu )); then install_cpu_seeker; fi
  install_self; save_conf; write_launcher_worker; write_unit
  svc start
  ok "worker updated"
}

# ---- summaries ------------------------------------------------------------------------------------------------------------
pool_join_command() { # the line to run on a worker
  local ip port="${POOL_PORT:-$(pool_server_port)}" tok="${POOL_TOKEN:-$(profile_pool_get token)}" pass=""
  ip=$(public_ip || true); ip="${ip:-SERVER_IP}"
  [[ -f "$ENV_FILE" ]] && pass='PROXIMA_KEY_PASSPHRASE=<passphrase of the key> '
  printf '%sbash proxima-gpu-miner.sh install --pool-worker http://%s:%s --pool-token %s --key-file ~/proxima.key%s -y' \
    "$pass" "$ip" "$port" "$tok" "$([[ "$BRANCH" != master ]] && echo " --branch $BRANCH")"
}
pool_server_port() { local l; l=$(profile_pool_get listen); l="${l##*:}"; echo "${l:-$POOL_DEFAULT_PORT}"; }
pool_server_bind() { local l; l=$(profile_pool_get listen); l="${l%:*}"; echo "${l:-0.0.0.0}"; }

summary_pool_server() {
  step "Pool server is set up"
  cat <<EOT
  ${C_B}How your workers join${C_0}   (one command per worker VPS; ${C_D}a worker installs no node and no database${C_0})
    1. copy the wallet key to the worker:   scp $WALLET_DIR/proxima.key USER@WORKER:~/proxima.key
    2. on the worker (needs git, curl and — for GPUs — the NVIDIA driver):
         $(pool_join_command)
    3. watch the combined hashrate here:    ${C_C}${SELF_SHORT:-$0}${C_0} status      members and totals:  ${C_C}${SELF_SHORT:-$0}${C_0} pool

  ${C_B}Good to know${C_0}
    * job server: ${C_W}${POOL_BIND:-0.0.0.0}:${POOL_PORT:-$POOL_DEFAULT_PORT}${C_0}, token ${C_W}${POOL_TOKEN}${C_0}  (also in $WALLET_DIR/proxi.yaml; new one: $0 pool rotate)
    * every worker holds the wallet key in its memory — only machines you trust. The token goes over plain HTTP:
      restrict the port to your workers' IPs (firewall / --pool-allow), or tunnel it (ssh -N -L ${POOL_PORT:-$POOL_DEFAULT_PORT}:127.0.0.1:${POOL_PORT:-$POOL_DEFAULT_PORT} server).
    * the pool adds hashrates up; the chance of finding a block is that of the TOTAL hashrate. It does not make blocks
      easier, it makes your VPS one miner. The reward goes to the one key (no payout sharing between different people).
    * if this server is down the workers idle (they reconnect by themselves when it is back).
EOT
}
summary_worker() {
  step "Done — pool worker is set up"
  cat <<EOT
  ${C_B}What runs${C_0}
    searches for  : ${C_W}$POOL_URL${C_0}   as '${POOL_NAME}'
    GPU           : $(if [[ "${USE_GPU:-0}" == 1 ]]; then echo "${C_G}yes${C_0} ($([[ -n "$GPU_DEVICES" ]] && echo "device(s) $GPU_DEVICES" || echo "all $GPU_N"), intensity $GPU_INTENSITY)"; else echo "no"; fi)
    CPU threads   : ${MINER_WORKERS:-0}
    service       : $(if unit_installed; then echo "systemd: $MINER_SVC"; else echo "background process (pid file $PID_FILE)"; fi)

  ${C_B}Everyday commands${C_0}   (shortcut: ${C_C}${SELF_SHORT:-$0}${C_0} <command>)
    status | watch   this worker + link to the pool     logs   follow it     pool   link details
    gpu [on|off|test]   GPU setup / self-test     bench   measure the GPU     update   rebuild     uninstall

  The earnings arrive at the key's wallet on the pool server — look there (status on the server shows balance + members).
EOT
}

# ---- dashboards ------------------------------------------------------------------------------------------------------------
# seconds from a Go duration like "1h2m3s", "5s", "250ms"
dur_secs() { awk -v d="$1" 'BEGIN{ s=0; while (match(d,/^[0-9.]+(h|ms|m|s)/)) { t=substr(d,RSTART,RLENGTH); if (t ~ /ms$/) {v=t+0; s+=v/1000} else if (t ~ /h$/) s+=(t+0)*3600; else if (t ~ /m$/) s+=(t+0)*60; else s+=t+0; d=substr(d,RLENGTH+1) } printf "%d", s }'; }

dash_pool_server() {
  pool_is_server || return 0
  sect "POOL — all members mine for this miner, under one key"
  local listen token; listen=$(profile_pool_get listen); token=$(profile_pool_get token)
  kv "job server" "${C_W}${listen:-?}${C_0}   ${C_D}$([[ -n "$token" ]] && echo "token-protected" || echo "NO token")${C_0}"
  local log line; log=$(miner_log); line=$(grep -F '   seekers: ' <<<"$log" | tail -n1 || true)
  if [[ -z "$line" || "$line" == *"no seeker has reported yet"* ]]; then
    # the proxi miner prints its per-member list only after one of ITS transits is confirmed, so it is empty
    # for a long time on a fresh pool. Show what can be seen without it: live TCP connections to the job port.
    local port conns nconn=0 ips=""; port=$(pool_server_port)
    conns=$( { ss -Htn state established "( sport = :$port )" 2>/dev/null || sudo -n ss -Htn state established "( sport = :$port )" 2>/dev/null; } | awk '{print $NF}' | sed -E 's/:[0-9]+$//; s/^\[(.*)\]$/\1/' | sort -u || true)
    if [[ -n "$conns" ]]; then nconn=$(wc -l <<<"$conns" | tr -d ' '); ips=$(tr '\n' ' ' <<<"$conns"); fi
    if (( nconn > 0 )); then
      kv "members" "${C_B}${C_G}${nconn}${C_0} ${C_D}worker machine(s) connected right now:${C_0} ${C_W}${ips}${C_0}"
      note "                    per-member hashrate appears after this miner's first confirmed transit (the miner only prints it then)"
      note "                    until then the hashrate line above is already the WHOLE pool, workers included"
    else
      kv "members" "${C_Y}no worker connected${C_0} ${C_D}(nothing is connected to port $port; add one with: $0 pool)${C_0}"
    fi
    return 0
  fi
  line="${line#*seekers: }"
  if [[ "$line" == "no seeker has reported yet"* ]]; then
    kv "members" "${C_Y}none connected yet${C_0} ${C_D}(no worker has reported; join one with: $0 pool)${C_0}"; return 0
  fi
  local rows="" total=0 nmem=0 nlive=0 e nm rt age secs
  # "name 1_234 H/s (seen 3s ago), name2 ..." — one member per entry
  while IFS= read -r e; do
    [[ -n "$e" ]] || continue
    if [[ "$e" =~ ^[[:space:]]*(.+)[[:space:]]([0-9_]+)[[:space:]]H/s[[:space:]]\(seen[[:space:]]([^[:space:]]+)[[:space:]]ago\) ]]; then
      nm="${BASH_REMATCH[1]}"; rt="${BASH_REMATCH[2]//_/}"; age="${BASH_REMATCH[3]}"; secs=$(dur_secs "$age")
      nmem=$((nmem + 1))
      if (( secs <= 60 )); then nlive=$((nlive + 1)); total=$((total + rt)); fi
      rows+="$nm|$rt|$age|$secs"$'\n'
    fi
  done < <(sed 's/), /)\n/g' <<<"$line")
  local k="" re='K=([0-9]+)' ; if [[ "$(grep -F 'totals:' <<<"$log" | tail -n1)" =~ $re ]]; then k=${BASH_REMATCH[1]}; fi
  [[ -n "$k" ]] || k=$(grep -oE 'difficulty K=[0-9]+' <<<"$log" | tail -n1 | cut -d= -f2 || true)
  local day=0
  if [[ "$k" =~ ^[0-9]+$ ]] && (( total > 0 )); then
    day=$(awk -v r="$total" -v k="$k" -v s="${NET_SLOT_SEC:-10.24}" -v rw="${NET_REWARD:-95000000}" -v u="${NET_UNIT:-1000000}" 'BEGIN{ p=1-exp(-r*s/2^k); printf "%.3f", p*86400/s*rw/u }') || day=0
  fi
  kv "members" "${C_B}${C_G}${nlive}${C_0} ${C_D}live of ${nmem} seen${C_0}   ${C_D}pool total (seekers)${C_0} ${C_B}${C_G}$(fmt_rate "$total")${C_0}${day:+   ${C_D}≈ ${C_G}${day} ${NET_TICKER:-PROX}/day${C_0}${C_D} best case for the whole pool${C_0}}"
  local nm2 rt2 age2 sec2 sh dot col
  while IFS='|' read -r nm2 rt2 age2 sec2; do
    [[ -n "$nm2" ]] || continue
    if (( sec2 <= 60 )); then dot=$(dot_on); col=$C_W; else dot=$(dot_off); col=$C_D; fi
    sh=""; if (( total > 0 && sec2 <= 60 )); then sh=$(awk -v a="$rt2" -v t="$total" 'BEGIN{printf "%.0f%%", a*100/t}'); fi
    printf '  %s%-17s%s %s %s%-24s%s %s%12s%s  %s%5s  seen %s ago%s\n' "$C_D" "" "$C_0" "$dot" "$col" "${nm2:0:24}" "$C_0" "$C_B$C_G" "$(fmt_rate "$rt2")" "$C_0" "$C_D" "$sh" "$age2" "$C_0"
  done <<<"$rows"
  note "                    'seen' is as of the miner's last list; a member not seen for 60 s is shown as gone (○) and not counted"
  note "                    each worker searches its own random nonce range, the best result of the round wins: one transit per round"
  return 0
}

dash_pool_worker() {
  sect "POOL WORKER — searching for the pool server"
  local st pid; st=$(svc_state); pid=$(svc_pid)
  kv "service" "$(svc_badge "$st")   ${C_D}$(proc_stats "$pid")${C_0}"
  kv "pool server" "${C_W}${POOL_URL}${C_0}   ${C_D}as '${POOL_NAME:-?}'${C_0}"
  pool_probe "$POOL_URL" "$POOL_TOKEN" || true
  POOL_K=""
  case "$PP_CODE" in
    200)
      kv "link" "$(dot_on) ${C_G}connected${C_0}   ${C_D}round trip ${PP_RTT} ms · token accepted${C_0}"
      local id k slot ttl; id=$(jq -r '.id // 0' <<<"$PP_JOB" 2>/dev/null || echo 0)
      if [[ "$id" != 0 ]]; then
        k=$(jq -r '.k // empty' <<<"$PP_JOB" 2>/dev/null || true); slot=$(jq -r '.slot // empty' <<<"$PP_JOB" 2>/dev/null || true); ttl=$(jq -r '.ttl_ms // empty' <<<"$PP_JOB" 2>/dev/null || true)
        POOL_K="$k"
        kv "current job" "${C_W}slot ${slot:-?}${C_0}  ${C_W}K=${k:-?}${C_0} ${C_D}(a nonce needs ${k:-?} trailing zero bits) · ${ttl:+$((ttl / 1000)) s left}${C_0}"
      else kv "current job" "${C_Y}none right now${C_0} ${C_D}(the server is between rounds, or its node is not synced)${C_0}"; fi
      pool_key_match
      case "$POOL_KEYMATCH" in
        yes) kv "key" "${C_G}the server mines under this worker's key${C_0}";;
        no)  kv "key" "${C_R}✘ DIFFERENT key${C_0} ${C_D}— the server refuses this worker's results. Install the pool's proxima.key here.${C_0}";;
        *)   kv "key" "${C_D}(checked while the server has a job)${C_0}";;
      esac;;
    401) kv "link" "$(dot_off) ${C_R}the server refuses the token${C_0}   ${C_D}(rotated? get the new one from the server: pool, then re-run install --pool-worker)${C_0}";;
    0)   kv "link" "$(dot_off) ${C_R}no answer${C_0}   ${C_D}(miner on the server stopped, wrong port, or its firewall does not let this VPS in)${C_0}"
         note "the seekers retry by themselves; check from here:  $0 check";;
    *)   kv "link" "$(dot_mid) ${C_Y}HTTP $PP_CODE${C_0}";;
  esac
  local log; log=$(miner_log)
  # CPU seeker: its last "job N: R H/s" line
  if [[ "${MINER_WORKERS:-0}" =~ ^[1-9][0-9]*$ ]]; then
    local cr; cr=$(grep -F '[cpu-seeker]' <<<"$log" | grep -oE ': [0-9]+ H/s' | tail -n1 | grep -oE '[0-9]+' || true)
    if [[ -n "$cr" ]]; then kv "CPU seeker" "${C_B}${C_G}$(fmt_rate "$cr")${C_0}   ${C_D}${MINER_WORKERS} thread(s)${C_0}"
    else kv "CPU seeker" "${C_Y}no rate yet${C_0} ${C_D}(${MINER_WORKERS} thread(s); reports every 10 s while the server has a job)${C_0}"; fi
    local cl; cl=$(grep -F '[cpu-seeker]' <<<"$log" | grep -E 'cannot|REFUS|REJECT' | tail -n1 || true)
    [[ -z "$cl" ]] || kv "last problem" "${C_Y}${cl:0:90}${C_0}"
  fi
}
POOL_K=""

# the `pool` command
cmd_pool() {
  load_conf
  local sub="${ARGS[0]:-show}"
  if pool_is_worker; then
    load_install
    step "Pool worker"
    dash_pool_worker
    note "to point this worker at another server or key: $0 install --pool-worker URL --pool-token T --key-file F"
    return 0
  fi
  if ! pool_is_server; then
    cat <<EOT
  This machine mines on its own. To combine several VPS into ONE miner:
    on the machine that should run the miner (the server):  $0 install --pool-server
    on every other VPS (a worker):                          $0 install --pool-worker http://SERVER:$POOL_DEFAULT_PORT --pool-token TOKEN --key-file ~/proxima.key
  (the server prints the exact worker command at the end of its install)
EOT
    return 0
  fi
  load_install
  case "$sub" in
    show|"")
      step "Pool server"
      kv "job server" "${C_W}$(profile_pool_get listen)${C_0}"
      kv "token" "${C_W}$(profile_pool_get token)${C_0}"
      printf '\n  %sJoin command for a worker%s (after copying proxima.key to it):\n    %s\n\n' "$C_B" "$C_0" "$(pool_join_command)"
      dash_pool_server
      note "tunnel instead of an open port:  on the worker  ssh -N -L ${POOL_PORT:-$(pool_server_port)}:127.0.0.1:${POOL_PORT:-$(pool_server_port)} USER@SERVER   and use --pool-worker http://127.0.0.1:$(pool_server_port)"
      note "(set --pool-bind 127.0.0.1 on the server to accept only tunnelled workers)";;
    join) pool_join_command; echo;;
    rotate)
      POOL_TOKEN=$(pool_new_token); POOL_PORT=$(pool_server_port); POOL_BIND=$(pool_server_bind)
      svc stop 2>/dev/null || true
      profile_pool_set "$POOL_BIND:$POOL_PORT" "$POOL_TOKEN"; save_conf
      svc start
      ok "new token: $POOL_TOKEN"
      warn "every worker is locked out until it is re-installed with the new token:  $(pool_join_command)";;
    *) die "usage: $0 pool [show|join|rotate]";;
  esac
}

# 'check' on a worker: link to the pool + GPU
cmd_check_worker() {
  load_install
  step "Connectivity: this VPS -> pool server $POOL_URL"
  local hp="${POOL_URL#*//}" host port; host="${hp%%:*}"; port="${hp##*:}"
  if timeout 6 bash -c ":</dev/tcp/$host/$port" 2>/dev/null; then ok "TCP $host:$port is open"
  else warn "TCP $host:$port does not answer — the server's firewall must allow this VPS ($(public_ip || true)) on port $port"; fi
  pool_probe "$POOL_URL" "$POOL_TOKEN" || true
  case "$PP_CODE" in
    200) ok "the pool server answers in ${PP_RTT} ms and accepts the token"; pool_key_match
         case "$POOL_KEYMATCH" in yes) ok "the server mines under this worker's key";; no) warn "the server mines under ANOTHER key than this worker's proxima.key";; esac
         [[ "$(jq -r '.id // 0' <<<"$PP_JOB" 2>/dev/null)" != 0 ]] || log "the server has no job at this moment (between rounds, or its node is not synced)";;
    401) warn "the server refuses the token";;
    0)   warn "no answer from $POOL_URL/seeker/job — the miner on the server may be stopped";;
    *)   warn "unexpected HTTP $PP_CODE";;
  esac
  if (( PP_RTT > 400 )) 2>/dev/null; then warn "round trip ${PP_RTT} ms is high: a far-away worker finds its nonces later (jobs last ~10 s); a server close to the node and workers helps"; fi
  step "GPU"
  if gpu_report; then
    if seeker_installed; then
      local rc=0; gpu_selftest --check || rc=$?
      if (( rc == 0 )); then ok "GPU seeker installed, kernel agrees with the CPU reference"; else warn "GPU self-test failed (exit $rc)"; fi
    else log "the GPU seeker is not installed"; fi
  fi
}

# >>> gpu-seeker payload (generated by gpu-seeker/tools/embed.py - do not edit by hand)
gpu_payload() {
  cat <<'GPU_SEEKER_PAYLOAD_EOF'
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
n38+/3z++fzz+efzz/8PP/8Jsk9dKQAwAgA=
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
  start|stop|restart) load_install; cmd_startstop "$CMD";;
  node-reset)      cmd_node_reset;;
  pool)            cmd_pool;;
  set-node)        cmd_set_node;;
  key)             cmd_key;;
  update)          cmd_update;;
  uninstall)       cmd_uninstall;;
  help|*)          usage;;
esac
