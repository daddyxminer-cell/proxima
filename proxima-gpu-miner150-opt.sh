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
#  Seeker 1.4.3: reusable HTTP report connections, delivery latency/verdict counters,
#  one-second reporting cadence and token-safe `pool diagnose` (own wins vs observed transits).
#  Retains live-path autotuning, amortized cancellation polls and RFC-vector startup checks.
#  Optional R/w2d, R/w2db3, R/w3d, R/w4d variants remove two carry chains per scalar doubling.
#  Startup validates and times them against the original kernels; no measured speed claim.
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
#                        Lazy-doubling candidates: R/w2d R/w2db3 R/w3d R/w4d (auto-tested with originals).
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

SCRIPT_VERSION="1.4.3"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
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
      log "  register kernels R (w2 / w2b3 / w3 / w4 + lazy-doubling d variants) …"
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
nkXpm8VhFWjihQtd+77f/B73+AFclPVFglF08Xt6uXMxPKyK7HfLW4Qbjwl/dbX1Br2Sf0042sHn
gO7WntCz2wvgN6FB7XpVVGKubvf6q3LLq9I/Jtdflb539avxjlrFWq1bGpzsbnByTYMp2nhXi3Fv
V5Orzu5ucnJNk+W2NodbqIMJriakhmLJYi23UqvHtjelVfwX7Z1eO4Qo2n30Qxuul+bbchMKrmam
Q8FbaElcNamtiZGbr26j4Mg/JtdfjZpm76TgqgZxDfEnuxucXNPgrRQc1dXubPJuCq7qSK5pskyu
oP5NCq7of3+xQ7xvshWpcmq9aKkO7DwMXnG0SnO33qbDD/15So+d45+/oFuvpPGRLW1kjTfq3Y49
lP3JtNYugPenx39aIX7+/Wc+aHT1+y9rv6tDRysB3izYU78FlFvWW7XxE10v3Lpmv3CNPGkiGAwa
zjv/67H2zhp9zH3jFqGZzWpmRwtunm+5Wb9JY8FNxz/tGdTHGCG6il4YX9/KiXw3qBrtHWEcgLTa
WkF2PkIX53Rya717cd4QjXf2VG3mj7YTqGouf3SvU0vp34agSIucErilrVanY87yT9YiYpT1tXs8
BW2D0nRHIBW7VimCqhs/dbcxA5D7tfIsU385gqqVzKpxMx9TcMorcuzyOW/VIcqGArH8vo7B4u6V
ub7rGjkGbRWZRp7i0QW5f1d7Tuojk/05bp1za3txVsXa1aXbjrkjmCu03xvWqi0JNXmuNrjxKph1
CHP1o6Hc77wuTgNeZZr23Wxo+PL8/WFwOcU/Dw4pgefDw+DRYfAYxHwY/L46uLEhWH54O7lyEeu3
iFpRLL9W00Hwu+A7oqDgLn1OO89+XjWl++y0eva88+zDagV8R6uhfeNRtWq+8wTZav5j7uGDbsOf
tC6yMtDQ6w+rN7/gUXnYb7lhf99c+tE7LX5Y3es87ktoPb5q6owYF411u/0zGkQa+qZJ9ZQ1D7zi
t5503/qLn7POGpTTG5F8a6cUm8mdtbSUxmwPX91bHdA8D15to07ebnkFYW4jwIbsVjQHAsCMfL5C
Qg+6zPM7IhTQWuvCw4pLrubgIRf6sDvZcy6oerl6Fdfm3acefO5XQ3V1w7LU0FTs6ed7am2bIz7h
yamhE+3efwTG97m3XT7Cz6MHAUWmBUe9B7c/76+Eti/ycTUA7SJx7fGqyN6fbv+5/1dSkh5wwa0C
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
