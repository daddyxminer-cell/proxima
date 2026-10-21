#!/usr/bin/env bash
# =============================================================================
#  proxima-gpu-miner.sh — standalone CPU + NVIDIA GPU Proxima miner for a VPS / rig
#
#  Mines on THIS machine and talks to a node that already runs (and is synced)
#  somewhere else.  No node, no database: it builds `proxi` from
#  https://github.com/lunfardo314/proxima (master) and runs `proxi node mine --seeker`
#  with a CUDA nonce seeker (`gpu_seeker`, sources shipped inside this script).
#  Target network: Proxima "take 1" (v1.1.0-take1 or newer, miner version 5).
#  Upgrade 1 activates at slot 29000: proxi selects the library/version for each slot
#  and stamps v5 transits at tick 83 (settlement tick 86). Submit solutions immediately;
#  these are transaction timestamps, NOT a reason to delay sending a solution.
#  Seeker 1.5.0: latency-aware settlement-tail sizing and TTL-bounded result requests.
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
#    (environment only)  GPU_KERNELS="r 2 1 0"  which kernel images to compile: r = the register-resident kernels
#                                        (R/w2 R/w2b3 R/w3 R/w4: no local-memory working set, the fast path),
#                                        2 1 0 = the legacy code-size levels (L2 smallest, L0 fully inlined).  The
#                                        seeker times every variant on each card and keeps the fastest.  "r 2" is a
#                                        short build that needs little RAM.
#                        GPU_VARIANT=R/w2    force one kernel instead of timing them (R/w2 R/w2b3 R/w3 R/w4, or
#                                        L0 L1 L2 optionally with /b4 /b6 /np)
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

SCRIPT_VERSION="1.5.0"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
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
  ok "source supports miner versions $minv..$mv; per-slot timestamps and TxVersion enabled"
}
check_node_policy() {
  [[ -n "$NODE_URL" ]] || return 0
  local si slot lc required minv mv reason=""
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
  local lv lvs="${GPU_KERNELS:-r 2 1 0}" suffix built=() lflags
  log "nvcc: sm_${archs// /, sm_} + PTX compute_$min, kernel levels: $lvs${ccb:+  (host compiler $ccb)}  — 1 to 5 minutes"
  for lv in $lvs; do
    [[ "$lv" =~ ^[012rR]$ ]] || { warn "ignoring unknown kernel level '$lv' in GPU_KERNELS"; continue; }
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
    / net: reports=[0-9]+/ {
      net=""
      allowed=" reports report_errors report_rtt_avg_ms report_rtt_max_ms results accepted beaten stale rejected result_errors result_rtt_avg_ms result_rtt_max_ms job_errors "
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
vn++f/4vDrQ2CwBwAwA=
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
