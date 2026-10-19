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
#                        GPU_VARIANT=L2      force one kernel instead of timing them (L0 L1 L2, optionally /b4 /b6 /np)
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
P/ef+8/95/5z/7n//Pd+/i/42AczAFgCAA==
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
