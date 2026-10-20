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
#  Seeker 1.4.1: nonblocking best-result submission, same-job beat/deadline updates,
#  one-second attempt reporting, blocking CUDA completion waits, and RFC-vector startup checks.
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

SCRIPT_VERSION="1.4.1"          # CPU+GPU edition (proxima-miner.sh is the CPU-only one)
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
  [[ -f "$SEEKER_FATBIN" || -f "${SEEKER_FATBIN%.fatbin}.r.fatbin" || -f "${SEEKER_FATBIN%.fatbin}.l2.fatbin" || -f "${SEEKER_FATBIN%.fatbin}.l1.fatbin" ]] || return 0
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
/bXr9b+Ns82HACADAA==
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
