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
