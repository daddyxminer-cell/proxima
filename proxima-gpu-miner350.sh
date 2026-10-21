#!/usr/bin/env bash
# =============================================================================
#  proxima-gpu-miner.sh — standalone CPU + NVIDIA GPU Proxima miner for a VPS / rig
#
#  Mines on THIS machine and talks to a node that already runs (and is synced)
#  somewhere else.  No node, no database: it builds `proxi` from
#  https://github.com/lunfardo314/proxima (master) and runs `proxi node mine --seeker`
#  with a CUDA nonce seeker (`gpu_seeker`, sources shipped inside this script).
#  Target network: Proxima "take 1" (node v1.1.1 or newer, miner version 5).
#  v1.1.1 fixes branch-state inconsistency at upgrade slot 29000; upgrade your node too.
#  Upgrade 1 activates at slot 29000: proxi selects the library/version for each slot
#  and stamps v5 transits at tick 83 (settlement tick 86). Submit solutions immediately;
#  these are transaction timestamps, NOT a reason to delay sending a solution.
#  Seeker 1.6.0: build L3; tune every resident grid per kernel, confirm the top three.
#  Retains latency-aware settlement-tail sizing and TTL-bounded result requests.
#  Keeps reusable HTTP connections, one-second reports and token-safe `pool diagnose`.
#  Retains live-path autotuning, amortized cancellation polls and RFC-vector startup checks.
#  CUDA only: this script does not provide an OpenCL backend or a measured hashrate guarantee.
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
#    pool diagnose read-only, token-safe log counters, raw-hit baseline and HTTP responsiveness (no restart)
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
#    (environment only)  GPU_KERNELS="r 3 2 1 0"  which kernel images to compile: r = the register-resident kernels
#                                        (R/w2 R/w2b3 R/w3 R/w4: no local-memory working set, the fast path),
#                                        3 2 1 0 = the legacy code-size levels (L2 smallest, L0 fully inlined).  The
#                                        seeker times every variant on each card and keeps the fastest.  "r 2" is a
#                                        short build that needs little RAM.
#                        GPU_VARIANT=R/w2    force one kernel instead of timing them (R/w2 R/w2b3 R/w3 R/w4, or
#                                        L0 L1 L2 L3 optionally with /b4 /b6 /np)
#                        GPU_SEEKER_PIPELINE=0  one launch at a time instead of the queued async path (for comparing)
#                        GPU_SEEKER_EARLY_ABORT=0  keep pipeline, disable host cancellation reads (benchmark A/B;
#                          old batches can finish after a job change, so leave enabled for normal mining)
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

SCRIPT_VERSION="1.6.0"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
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
GPU_VARIANT="${GPU_VARIANT:-}"           # force a kernel (R/w2 | R/w2b3 | R/w3 | R/w4 | L0 | L1 | L2 | with /b4 /b6); empty = time them, keep the fastest
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
  check_source_policy
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
src_min_miner_version() {
  grep -hoE 'minMinerVersion +uint32 += +[0-9]+' "$SRC_DIR/proxi/node_cmd/mine/mine_upgrade.go" 2>/dev/null | grep -oE '[0-9]+$' | head -1 || true
}
check_source_policy() {
  local mv minv
  mv=$(src_miner_version); minv=$(src_min_miner_version)
  [[ "$mv" =~ ^[0-9]+$ && "$minv" =~ ^[0-9]+$ ]] && (( mv >= 5 && minv <= 5 && minv <= mv )) \
    || die "source lacks miner-v5 upgrade support; use --branch v1.1.0-take1 (or a newer compatible release)"
  # Fail closed for old/custom trees. Do not patch consensus constants into a v4 miner:
  # the timestamp AND transaction library index must be selected for the target slot.
  grep -q 'MineSettlementTick() - c.TransactionPaceSequencer' "$SRC_DIR/proxi/node_cmd/mine/mine_upgrade.go" \
    && grep -q 'txbuildercore.New(slotConsts.UpgradeIndex)' "$SRC_DIR/proxi/node_cmd/mine/mine.go" \
    || die "source is missing the reviewed per-slot settlement/TxVersion builders; use --branch v1.1.0-take1"
  if [[ -n "${OWN_NODE:-}" ]] && grep -q 'ledger.HasPendingUpgradeForSlot' "$SRC_DIR/ledger/multistate/upgrade_inject.go" 2>/dev/null; then
    die "managed node source has the slot-29000 branch-state bug; update with --branch v1.1.1 or newer (do not reset the database blindly)"
  fi
  ok "source supports miner versions $minv..$mv; per-slot timestamps and TxVersion enabled"
}
check_node_policy() {
  [[ -n "$NODE_URL" ]] || return 0
  local si slot lc required minv mv ni version reason=""
  ni=$(curl -fs -m 5 "$NODE_URL/api/v1/node_info" 2>/dev/null || true)
  version=$(jq -r '.version // empty' <<<"$ni" 2>/dev/null || true)
  case "$version" in
    v1.1.0|v1.1.0-take1|v1.1.0p1)
      warn "your node is $version: v1.1.1 fixes branch-state inconsistency at slot 29000; upgrade the actual node, not only GPU workers";;
  esac
  si=$(sync_info "$NODE_URL")
  slot=$(jq -r '.current_slot // empty' <<<"$si" 2>/dev/null || true)
  if [[ ! "$slot" =~ ^[0-9]+$ ]]; then
    warn "cannot check your node's current-slot rules; compatibility is unverified"
    return 0
  fi
  # Query the node we actually mine through, not the first public witness. The
  # default constants can lag activation until the upgrade has a reliable proof.
  if ! lc=$(curl -fs -m 8 "$NODE_URL/api/v1/ledger_constants?slot=$slot" 2>/dev/null); then
    warn "your node did not return slot-$slot constants; compatibility is unverified"
    return 0
  fi
  mv=$(src_miner_version); minv=$(src_min_miner_version)
  required=$(jq -r '.miner_version // empty' <<<"$lc" 2>/dev/null || true)
  if ! jq -e 'has("upgrade_index") and has("upgrade_slot")' <<<"$lc" >/dev/null 2>&1; then
    reason="your node does not expose the library-upgrade metadata (old node/API)"
  elif [[ ! "$required" =~ ^[0-9]+$ ]] || (( required < minv || required > mv )); then
    reason="slot $slot requires miner version ${required:-unknown}, source supports $minv..$mv"
  fi
  if [[ -n "$reason" ]]; then
    if [[ -n "$OWN_NODE" ]]; then
      warn "$reason; this managed node must be upgraded with proxi before mining resumes"
    else
      die "$reason. Upgrade the remote node and proxi to v1.1.0-take1 or newer; updating a GPU worker alone is not enough"
    fi
  else
    ok "your node: slot $slot requires miner v$required (supported)"
  fi
}
minor_of() { echo "$1" | sed -E 's/^v?([0-9]+\.[0-9]+).*/\1/'; }
check_compat() {
  check_source_policy
  check_node_policy
  load_net_info || { warn "could not read network ledger info; public-witness compatibility is unverified"; return 0; }
  log "network ledger: ${C_B}${NET_DESC:-?}${C_0}  nodes ${NET_NODE_VER:-?}, served miner version ${NET_MINER_VER:-?}"
  if [[ -n "${SRC_VERSION:-}" && -n "$NET_NODE_VER" && "$(minor_of "$SRC_VERSION")" != "$(minor_of "$NET_NODE_VER")" ]]; then
    warn "source is $SRC_VERSION but the witness runs $NET_NODE_VER; use a network-compatible release (upgrade 1: v1.1.0-take1)"
  fi
  local mv minv
  mv=$(src_miner_version); minv=$(src_min_miner_version)
  if [[ "$NET_MINER_VER" =~ ^[0-9]+$ ]] && (( NET_MINER_VER < minv || NET_MINER_VER > mv )); then
    die "network requires miner v$NET_MINER_VER; source supports $minv..$mv — update proxi before mining"
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
  # status/update can reach this without choose_gpu/gpu_report. Do not compare a
  # single-card build with the broad no-detection fallback just because arrays are empty.
  if (( ${#GPU_CCS[@]} == 0 )); then detect_gpus; fi
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
gpu_stale_reason() { # 0 + reason = rebuild needed; 1 = matching local source/build requirements
  [[ -x "$SEEKER_BIN" ]] || { printf 'GPU seeker binary missing'; return 0; }
  [[ -f "$GPU_INFO" ]] || { printf 'GPU build metadata missing'; return 0; }
  compgen -G "${SEEKER_FATBIN%.fatbin}*.fatbin" >/dev/null || { printf 'GPU kernel images missing'; return 0; }
  local saved payload source archs current wanted a
  saved=$(head -n1 "$GPU_INFO" 2>/dev/null || true)
  payload=$(sed -nE 's/^payload=([0-9a-f]+) proxima=.*$/\1/p' <<<"$saved")
  source=$(sed -nE 's/^payload=[0-9a-f]+ proxima=([0-9a-f]+) archs=.*$/\1/p' <<<"$saved")
  archs="${saved##*archs=}"
  [[ -n "$payload" && -n "$source" && "$archs" =~ ^[0-9]+([[:space:]][0-9]+)*$ ]] || { printf 'GPU build metadata is unrecognized'; return 0; }
  [[ "$payload" == "$(gpu_payload_hash)" ]] || { printf 'embedded seeker sources differ from installed build'; return 0; }
  current=$(as_user git -C "$SRC_DIR" rev-parse --short=10 HEAD 2>/dev/null || true)
  [[ -n "$current" ]] || { printf 'local Proxima source revision cannot be checked'; return 0; }
  [[ "$source" == "$current" ]] || { printf 'local Proxima source revision changed (%s -> %s)' "$source" "$current"; return 0; }
  if [[ -z "$GPU_ARCHS" ]] && (( ${#GPU_CCS[@]} == 0 )); then detect_gpus; fi
  # If detection is unavailable, do not invent a mismatch using build-time fallback architectures.
  if [[ -n "$GPU_ARCHS" ]] || (( ${#GPU_CCS[@]} > 0 )); then
    wanted=$(gpu_arch_list)
    if [[ -n "$GPU_ARCHS" ]]; then
      [[ "$(tr ' ' '\n' <<<"$archs" | sort -nu | xargs)" == "$(tr ' ' '\n' <<<"$wanted" | sort -nu | xargs)" ]] || { printf 'explicit CUDA architecture configuration changed'; return 0; }
    else
      # A multi-architecture image is current when it already contains all visible cards.
      for a in $wanted; do
        [[ " $archs " == *" $a "* ]] || { printf 'native CUDA architecture sm_%s is absent from installed build' "$a"; return 0; }
      done
    fi
  fi
  return 1
}
gpu_stale() { gpu_stale_reason >/dev/null; }
gpu_update_notice() {
  local reason
  if reason=$(gpu_stale_reason 2>/dev/null); then
    kv "" "${C_Y}GPU rebuild needed: $reason — run: $0 update${C_0}"
  fi
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
  # "r" = the register-resident kernels (gpu-seeker/cuda/vrf_fast.h: no local-memory working set, wNAF widths 2/3/4);
  # it is built first, so it is there even when a bigger legacy level runs out of memory.
  local lv lvs="${GPU_KERNELS:-r 3 2 1 0}" suffix built=() lflags
  log "nvcc: sm_${archs// /, sm_} + PTX compute_$min, kernel levels: $lvs${ccb:+  (host compiler $ccb)}  — 1 to 5 minutes"
  for lv in $lvs; do
    [[ "$lv" =~ ^[0123rR]$ ]] || { warn "ignoring unknown kernel level '$lv' in GPU_KERNELS"; continue; }
    if [[ "$lv" == [rR] ]]; then
      lv=R; suffix=".r"; lflags=(-DVRF_CALL_LEVEL=1 -DVRF_FAST -DVRF_FAST_ONLY)
      log "  register kernels R (w2 / w2b3 / w3 / w4) …"
    else
      suffix=""; [[ "$lv" != 0 ]] && suffix=".l$lv"; lflags=(-DVRF_CALL_LEVEL="$lv")
      log "  kernel level L$lv …"
    fi
    if ! ( cd "$GPU_SRC/cuda" && as_user env PATH="$(dirname "$NVCC"):$PATH" "$NVCC" -U_GNU_SOURCE -O3 -std=c++14 --fatbin "${lflags[@]}" \
             "${gen[@]}" ${ccb:+-ccbin "$ccb"} -o "$GPU_BUILD/gpu_seeker$suffix.fatbin.new" kernel.cu ); then
      if (( ${#built[@]} > 0 )); then warn "level L$lv failed to compile — continuing with the levels that did (${built[*]})"; continue; fi
      die "nvcc failed. Typical causes: a host gcc newer than 14 (install g++-13 or pass GPU_CCBIN=/usr/bin/g++-13), out of memory (add swap), or an architecture this nvcc does not know (--gpu-arch \"86 89\"; sm_120 needs CUDA >= 12.8)"
    fi
    if [[ "$lv" == R ]]; then built+=("R"); else built+=("L$lv"); fi
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
  as_user rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}".{r,l1,l2,l3}.fatbin 2>/dev/null || as_root rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}".{r,l1,l2,l3}.fatbin
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
  # Pass diagnostic controls explicitly through sudo and into generated launchers.
  local name value
  for name in GPU_SEEKER_PIPELINE GPU_SEEKER_EARLY_ABORT; do
    value="${!name:-1}"
    [[ "$value" == 0 || "$value" == 1 ]] || die "$name must be 0 or 1"
    SEEKER_ENV+=("$name=$value")
  done
  return 0
}
random_seed() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# gpu_selftest [--list|--check|--bench N]: check on a throw-away key; BENCHMARK on the installed key.
# The secret's wNAF schedule affects timing, so a different random key is not a controlled A/B comparison.
# The benchmark remains entirely local and never sends the key or any work to a server.
gpu_selftest() {
  local mode="${1:---check}"; shift || true
  [[ -x "$SEEKER_BIN" ]] || { warn "gpu_seeker is not installed ($SEEKER_BIN)"; return 3; }
  seeker_env
  local out rc=0
  if [[ "$mode" == --bench ]]; then
    [[ -f "$WALLET_DIR/proxima.key" ]] || { warn "benchmark requires the installed mining key: $WALLET_DIR/proxima.key"; return 1; }
    log "benchmark uses the installed mining key and live execution path (no network submissions)"
    out=$(cd "$WALLET_DIR" && as_user env "${SEEKER_ENV[@]}" PROXIMA_KEY_PASSPHRASE="${PROXIMA_KEY_PASSPHRASE:-}" \
      timeout 600 "$SEEKER_BIN" --key-file "$WALLET_DIR/proxima.key" "$mode" "$@" 2>&1 </dev/null) || rc=$?
  else
    out=$(as_user env "${SEEKER_ENV[@]}" NONCE_SEEKER_SEED="$(random_seed)" timeout 600 "$SEEKER_BIN" "$mode" "$@" 2>&1 </dev/null) || rc=$?
  fi
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
      printf '    low     short kernels: the desktop stays smooth, lower hashrate\n    normal  ~150 ms kernels (recommended for a dedicated rig / VPS)\n    high    ~300 ms kernels, same resident wave, fewer launches (benchmark before choosing)\n'
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
    gpu_update_notice
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
  if [[ "${USE_GPU:-0}" == 1 ]]; then gpu_update_notice; fi
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
  step "GPU benchmark (sampling budget $secs s per GPU; minimum 2 s per variant, plus warm-up)"
  local was; was=$(svc_state)
  case "$was" in active|running) log "pausing the miner for the measurement"; svc stop 2>/dev/null || true;; *) was="";; esac
  local rc=0
  gpu_selftest --bench "$secs" || rc=$?
  if [[ -n "$was" ]]; then svc start || true; fi
  (( rc == 0 )) || die "benchmark failed (exit $rc)"
  log "live execution path, installed mining key, intensity '${GPU_INTENSITY:-normal}'; real wall rate also includes no-job time and job changes"
  log "A/B cancellation reads: repeat with GPU_SEEKER_EARLY_ABORT=0 (pipeline unchanged); do not compare different keys or GPU clock/power settings"
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
    warn "Upgrade 1 is NOT a network reset. If the node refuses late upgrade, inspect logs node and restore a trusted snapshot; do not wipe the database automatically."
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
  as_root rm -f "$BIN_DIR/proxi" "$BIN_DIR/proxi.prev" "$BIN_DIR/proxima" "$BIN_DIR/proxima.prev" "$SEEKER_BIN" "$CPU_SEEKER_BIN" "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}".{r,l1,l2,l3}.fatbin
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

# Aggregate only known numeric fields: never echo raw logs, endpoints, names, keys or tokens.
# Counts describe the bounded log excerpt, not lifetime results or a closed accounting period.
pool_log_summary() {
  awk '
    /difficulty K=[0-9]+/ {
      s=$0; sub(/^.*difficulty K=/,"",s); sub(/[^0-9].*$/,"",s); k=s+0; have_k=1
    }
    /mining\.\.\.|no solution for slot|expected ~.*attempts per solution/ {
      s=$0; gsub(/_/,"",s)
      if (match(s,/[0-9]+ H\/s/)) { r=substr(s,RSTART,RLENGTH)+0; have_rate=1 }
    }
    /(SOLVED|IMPROVED) transit #[0-9]+; submitting/ { submitted++ }
    /transit #[0-9]+ .* accepted/ {
      if ($0 ~ /accepted \(own\)/) own_accepted++; else observed++
    }
    /confirmed transit #[0-9]+ .*\([0-9]+ of ours\)/ {
      s=$0; sub(/^.*\(/,"",s); sub(/ of ours\).*$/,"",s); own_confirmed+=s+0
    }
    /none of ours; re-anchoring/ { reanchors++ }
    /no solution for slot/ { empty_rounds++ }
    /REFUSING job/ { refused++ }
    /REJECTED nonce/ { rejected++ }
    /cannot reach the miner/ { job_errors++ }
    /cannot post the solution/ { result_errors++ }
    /cannot report attempts/ { report_errors++ }
    /the miner has moved on/ { stale++ }
    /seeker build [0-9]+[.][0-9]+[.][0-9]+;/ {
      s=$0; sub(/^.*seeker build /,"",s); sub(/;.*/,"",s)
      if (s ~ /^[0-9]+[.][0-9]+[.][0-9]+$/) build=s
    }
    /full reward topping up delegation/ { topups++ }
    /PROX into a new delegation at sequencer/ { newdelegations++ }
    /tuning selected variant=/ {
      s=$0
      if (match(s,/gpu[0-9]+ /)) {
        gpu=substr(s,RSTART,RLENGTH-1); fields=""
        for (i=1;i<=NF;i++) {
          if ($i ~ /^variant=(R\/w[234](b3)?|L[0123](\/(b4|b6|np))?)$/ || $i ~ /^(blocks|conservative_mhs)=[0-9]+([.][0-9]+)?$/) fields=fields " " $i
        }
        if (fields!="") tuning[gpu]=fields
      }
    }
    / net: reports=[0-9]+/ {
      net=""
      allowed=" reports report_errors report_rtt_avg_ms report_rtt_max_ms results accepted beaten stale rejected result_errors result_rtt_avg_ms result_rtt_max_ms job_errors delivery_margin_ms "
      for (i=1;i<=NF;i++) {
        if ($i !~ /^[a-z_]+=[0-9]+([.][0-9]+)?$/) continue
        split($i,a,"=")
        if (index(allowed," " a[1] " ")) net=net " " $i
      }
    }
    END {
      printf "  Log excerpt: local SOLVED/IMPROVED submissions=%d; accepted (own)=%d; confirmed own transits=%d\n", submitted,own_accepted,own_confirmed
      printf "  Unmarked observed accepted transits=%d (NOT an own-win counter); competitor re-anchors=%d; empty rounds=%d\n", observed,reanchors,empty_rounds
      printf "  Worker error lines: refused_jobs=%d rejected_results=%d stale_results=%d job_errors=%d result_errors=%d report_errors=%d\n", refused,rejected,stale,job_errors,result_errors,report_errors
      if (have_k && have_rate && r>0 && k<=512) {
        mean=2^k/r; p=100*(1-exp(-10/mean))
        printf "  Last coordinator rate=%.3f MH/s; K=%d; raw qualifying-nonce mean=%.1f s; chance in 10 s=%.2f%%\n",r/1e6,k,mean,p
        print "  Baseline only: constant rate/K, ignores beat, changing jobs, competitors and settlement. NOT expected wins or payout."
      } else print "  Raw-hit baseline unavailable: need coordinator difficulty and hashrate lines."
      if (build!="") print "  Seeker version observed in log: " build
      else print "  Seeker runtime version not present in log excerpt (script/fingerprint alone does not prove which process is running)."
      for (gpu in tuning) print "  Last confirmed tuning " gpu ":" tuning[gpu]
      printf "  Payout-choice log lines: new delegation=%d; full-reward top-up=%d (choices, NOT wins or paid amounts)\n",newdelegations,topups
      if (net!="") print "  Latest worker network counters (cumulative since seeker start):" net
      else print "  No worker network counters in this excerpt (run diagnose on updated GPU workers too)."
      print "  Excerpt counters may span restarts and omit older submissions/confirmations; do not divide them into a win ratio."
    }'
}

cmd_pool_diagnose() {
  load_install
  step "Pool diagnostics (read-only; no token, key, join command or raw log output)"
  printf '  Role: %s; script: %s; configured CPU mining threads: %s\n' "${POOL_ROLE:-solo}" "$SCRIPT_VERSION" "${MINER_WORKERS:-unknown}"
  local n pid logtext url token bind i reason good=0 times=""
  if [[ "${USE_GPU:-0}" == 1 ]]; then
    if reason=$(gpu_stale_reason); then printf '  Local GPU build: rebuild needed - %s\n' "$reason"
    else printf '  Local GPU build: source fingerprint matches (not a remote sequencer version check)\n'; fi
  fi
  n=$(nproc 2>/dev/null || echo 1)
  printf '  Available logical CPUs: %s; host load averages: ' "$n"
  awk '{print $1, $2, $3}' /proc/loadavg 2>/dev/null || true
  pid=$(svc_pid)
  if [[ "$pid" =~ ^[0-9]+$ && "$pid" != 0 ]]; then
    # No command-line arguments: they can contain the bearer token.
    ps -p "$pid" -o pid=,pcpu=,pmem=,ni=,nlwp=,etime= | awk '{printf "  Launcher/miner PID=%s CPU=%s%% memory=%s%% nice=%s threads=%s elapsed=%s (children not included)\n",$1,$2,$3,$4,$5,$6}' || true
  fi
  if [[ "${POOL_ROLE:-}" == server && "${MINER_WORKERS:-0}" =~ ^[0-9]+$ ]] && (( MINER_WORKERS >= n )); then
    printf '  Caution: CPU mining uses at least all visible CPUs. Measure scheduler pressure; consider reserving coordinator CPU capacity.\n'
  fi
  if [[ "${POOL_ROLE:-}" == worker ]]; then
    printf '  Worker cannot verify coordinator/node version, payout policy or confirmed wins; run diagnose on the coordinator too.\n'
  else
    local ni version
    ni=$(curl -fs -m 5 "$NODE_URL/api/v1/node_info" 2>/dev/null || true)
    version=$(jq -r '.version // empty' <<<"$ni" 2>/dev/null || true)
    if [[ "$version" =~ ^v?[0-9]+[.][0-9]+[.][0-9]+([a-zA-Z0-9.-]+)?$ ]]; then
      printf '  Mining node reports version: %s (not proof of correct branch state)\n' "$version"
      case "$version" in v1.1.0|v1.1.0-take1|v1.1.0p1) printf '  ACTION: upgrade this node to v1.1.1 or newer; known slot-29000 branch-state bug.\n';; esac
    else printf '  Mining node version: unavailable/unrecognized.\n'; fi
  fi
  logtext=$(miner_log)
  printf '%s\n' "$logtext" | pool_log_summary
  if [[ "${POOL_ROLE:-}" == worker ]]; then
    url="$POOL_URL"; token="$POOL_TOKEN"
  else
    bind=$(pool_server_bind)
    case "$bind" in 0.0.0.0|"") bind=127.0.0.1;; '[::]'|'::') bind='[::1]';; esac
    url="http://$bind:$(pool_server_port)"; token=$(profile_pool_get token)
  fi
  printf '  Five immediate job GET probes (not long-polls; a fresh connection per probe):\n'
  for i in 1 2 3 4 5; do
    pool_probe "$url" "$token"
    printf '    sample %s: HTTP %s, total_ms=%s\n' "$i" "$PP_CODE" "${PP_RTT:-unavailable}"
    if [[ "$PP_CODE" == 200 && "$PP_RTT" =~ ^[0-9]+$ ]]; then
      good=$((good + 1)); times+="$PP_RTT"$'\n'
    fi
    (( i == 5 )) || sleep 0.2
  done
  printf '  Successful probes: %s/5\n' "$good"
  if (( good > 0 )); then
    printf '%s' "$times" | awk 'NF {n++;s+=$1;if($1>m)m=$1} END {printf "  Successful GET total time: avg=%.1f ms, max=%.1f ms (connect + server + transfer, NOT isolated network RTT)\n",s/n,m}'
  fi
  printf '  GET probes do not measure solution verification or node/sequencer propagation. Compare server and worker diagnostics.\n'
  printf '  For CPU contention, capture vmstat 1 10; for clock sync, check timedatectl status. Do not send wallet files or tokens.\n'
}

# the `pool` command
cmd_pool() {
  load_conf
  local sub="${ARGS[0]:-show}"
  if [[ "$sub" == diagnose ]]; then cmd_pool_diagnose; return 0; fi
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
    *) die "usage: $0 pool [show|diagnose|join|rotate]";;
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
