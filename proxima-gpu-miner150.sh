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
P/D5/3Ih4PsAcAMA
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
