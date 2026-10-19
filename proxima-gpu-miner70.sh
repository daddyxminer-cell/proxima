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
#    (environment only)  GPU_KERNELS="3 2 1 0"  which kernel code-size levels to compile (L2 smallest, L3 = L2 with the point
#                                        operations inlined, L0 = fully
#                                        inlined; the seeker times them on each card and keeps the fastest. "2 1"
#                                        skips L0: a much shorter build, and needs less RAM)
#                        GPU_VARIANT=L2      force one kernel instead of timing them (L0 L1 L2 L3, optionally /b4 /b6 /np)
#                        GPU_SEEKER_PIPELINE=0  one launch at a time instead of the queued async path (for comparing)
#    OPTIONAL overclock / power / fan control (off unless you give a value; NVML, needs root, no X server):
#    --power W|N%|max   [OC_POWER]       GPU power limit in watts (clamped to the card's min..max), or % of its default
#    --core MHZ         [OC_CORE]        core clock OFFSET, e.g. 100 (the mining kernel is compute-bound: this is
#                                        what turns spare power into hashrate; -200 trades speed for efficiency)
#    --mem MHZ          [OC_MEM]         memory clock offset (Proxima's kernel barely touches VRAM: no gain, only
#                                        heat; a NEGATIVE offset saves power - normally leave this unset)
#    --lock-core MIN,MAX [OC_LOCK_CORE]  pin the core clock range in MHz (e.g. 2400,2700: steadier, often cooler)
#    --fan PCT|auto     [OC_FAN]         fixed fan speed 30..100 %, or `auto` (the card's own curve)
#    --oc-off                            forget all of the above and put the cards back
#                        Applied before the miner starts (systemd ExecStartPre) and restored when it stops, so a
#                        stopped miner leaves the card as it was; a reboot clears everything anyway.
#                        `oc [show|apply|reset]` looks at / applies / undoes it by hand;  `oc --power 450 --fan 70` sets it.
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
OC_POWER="${OC_POWER:-}"; OC_CORE="${OC_CORE:-}"; OC_MEM="${OC_MEM:-}"; OC_LOCK_CORE="${OC_LOCK_CORE:-}"; OC_FAN="${OC_FAN:-}"   # optional overclock add-on
OC_OFF=0
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
    --power)        OC_POWER="${2:?--power needs a value (watts, N%, or max)}"; shift 2;;
    --core)         OC_CORE="${2:?--core needs a clock offset in MHz}"; shift 2;;
    --mem)          OC_MEM="${2:?--mem needs a clock offset in MHz}"; shift 2;;
    --lock-core)    OC_LOCK_CORE="${2:?--lock-core needs MIN,MAX in MHz}"; shift 2;;
    --fan)          OC_FAN="${2:?--fan needs a percentage (30-100) or auto}"; shift 2;;
    --oc-off)       OC_OFF=1; OC_POWER=""; OC_CORE=""; OC_MEM=""; OC_LOCK_CORE=""; OC_FAN=""; shift;;
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
  if [[ "$OC_OFF" != 1 ]]; then
    [[ -n "$OC_POWER"     ]] || OC_POWER=$(conf_get OC_POWER)
    [[ -n "$OC_CORE"      ]] || OC_CORE=$(conf_get OC_CORE)
    [[ -n "$OC_MEM"       ]] || OC_MEM=$(conf_get OC_MEM)
    [[ -n "$OC_LOCK_CORE" ]] || OC_LOCK_CORE=$(conf_get OC_LOCK_CORE)
    [[ -n "$OC_FAN"       ]] || OC_FAN=$(conf_get OC_FAN)
  fi
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
  printf 'MINER_WORKERS=%s\nMINER_MAX_KHS=%s\nNO_PUBLIC=%s\nOWN_NODE=%s\nKEY_MODE=%s\nUSE_GPU=%s\nGPU_DEVICES=%s\nGPU_INTENSITY=%s\nGPU_ARCHS=%s\nPOOL_ROLE=%s\nPOOL_PORT=%s\nPOOL_BIND=%s\nPOOL_TOKEN=%s\nPOOL_URL=%s\nPOOL_NAME=%s\nOC_POWER=%s\nOC_CORE=%s\nOC_MEM=%s\nOC_LOCK_CORE=%s\nOC_FAN=%s\n' \
    "$MINER_WORKERS" "${MINER_MAX_KHS:-0}" "$([[ -n "$OWN_NODE" ]] && echo 0 || echo "${NO_PUBLIC:-0}")" "${OWN_NODE:-0}" "${KEY_MODE:-}" "${USE_GPU:-0}" "${GPU_DEVICES:-}" "${GPU_INTENSITY:-normal}" "${GPU_ARCHS:-}" \
    "${POOL_ROLE:-}" "${POOL_PORT:-}" "${POOL_BIND:-}" "${POOL_TOKEN:-}" "${POOL_URL:-}" "${POOL_NAME:-}" \
    "${OC_POWER:-}" "${OC_CORE:-}" "${OC_MEM:-}" "${OC_LOCK_CORE:-}" "${OC_FAN:-}" | as_user tee "$CONF_FILE" >/dev/null
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
  [[ -f "$SEEKER_FATBIN" || -f "${SEEKER_FATBIN%.fatbin}.l3.fatbin" || -f "${SEEKER_FATBIN%.fatbin}.l2.fatbin" || -f "${SEEKER_FATBIN%.fatbin}.l1.fatbin" ]] || return 0
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
  local lv lvs="${GPU_KERNELS:-3 2 1 0}" suffix built=()
  log "nvcc: sm_${archs// /, sm_} + PTX compute_$min, kernel levels: $lvs${ccb:+  (host compiler $ccb)}  — 1 to 5 minutes"
  for lv in $lvs; do
    [[ "$lv" =~ ^[0123]$ ]] || { warn "ignoring unknown kernel level '$lv' in GPU_KERNELS"; continue; }
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
  as_user rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin" "${SEEKER_FATBIN%.fatbin}.l3.fatbin" 2>/dev/null || as_root rm -f "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin" "${SEEKER_FATBIN%.fatbin}.l3.fatbin"
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

# =============================================================================
#  OPTIONAL overclock / power / fan control (--power --core --mem --lock-core --fan, `oc` command)
#  Off unless a value is set.  Done by gpu-seeker/tools/proxima-gpu-oc.py (NVML via ctypes), which is installed
#  ROOT-OWNED in /usr/local/lib and run by systemd as root before the miner starts (ExecStartPre) and after it stops
#  (ExecStopPost: the card goes back to what it had).  The miner itself never runs as root because of this.
# =============================================================================
OC_TOOL="${OC_TOOL:-/usr/local/lib/proxima-gpu-oc.py}"
OC_CONF="${OC_CONF:-/etc/proxima-gpu-oc.conf}"
oc_configured() { [[ -n "$OC_POWER$OC_CORE$OC_MEM$OC_LOCK_CORE$OC_FAN" ]]; }
oc_unit_lines() { # lines for the [Service] section ("" when nothing is configured)
  oc_configured || return 0
  printf 'ExecStartPre=-+%s apply --config %s\nExecStopPost=-+%s reset\n' "$OC_TOOL" "$OC_CONF" "$OC_TOOL"
}
oc_validate() {
  [[ -z "$OC_POWER" || "$OC_POWER" =~ ^([0-9]+(\.[0-9]+)?%?|max|default)$ ]] || die "bad --power '$OC_POWER' (watts, e.g. 450, a percentage of the default, e.g. 100%, or max)"
  [[ -z "$OC_CORE" || "$OC_CORE" =~ ^-?[0-9]+$ ]] || die "bad --core '$OC_CORE' (a clock offset in MHz, e.g. 100 or -150)"
  [[ -z "$OC_MEM" || "$OC_MEM" =~ ^-?[0-9]+$ ]] || die "bad --mem '$OC_MEM' (a clock offset in MHz, e.g. 500 or -300)"
  [[ -z "$OC_LOCK_CORE" || "$OC_LOCK_CORE" =~ ^[0-9]+,[0-9]+$ ]] || die "bad --lock-core '$OC_LOCK_CORE' (MIN,MAX in MHz, e.g. 2400,2700)"
  [[ -z "$OC_FAN" || "$OC_FAN" == auto || "$OC_FAN" =~ ^[0-9]+$ ]] || die "bad --fan '$OC_FAN' (30-100, or auto)"
  if [[ "$OC_FAN" =~ ^[0-9]+$ ]] && (( OC_FAN < 30 || OC_FAN > 100 )); then die "--fan must be 30..100 (or auto)"; fi
  return 0
}
oc_install_tool() { # root-owned copy, taken straight from the embedded payload
  command -v python3 >/dev/null 2>&1 || die "the overclock add-on needs python3 (apt install python3)"
  local tmp; tmp=$(mktemp)
  gpu_payload | base64 -d | tar -xzO tools/proxima-gpu-oc.py >"$tmp" 2>/dev/null && [[ -s "$tmp" ]] || { rm -f "$tmp"; die "the overclock tool is missing from this script's payload"; }
  as_root install -m 0755 -o root -g root "$tmp" "$OC_TOOL"; rm -f "$tmp"
}
oc_write_conf() {
  local tmp; tmp=$(mktemp)
  { printf '# written by proxima-gpu-miner.sh (--power/--core/--mem/--lock-core/--fan); edit with: proxima-gpu-miner.sh oc ...\n'
    printf 'devices=%s\npower=%s\ncore=%s\nmem=%s\nlock_core=%s\nfan=%s\n' "${GPU_DEVICES:-all}" "$OC_POWER" "$OC_CORE" "$OC_MEM" "$OC_LOCK_CORE" "$OC_FAN"; } >"$tmp"
  as_root install -m 0644 -o root -g root "$tmp" "$OC_CONF"; rm -f "$tmp"
}
oc_run() { as_root "$OC_TOOL" "$@"; }
oc_sync() { # called by write_unit, i.e. by install / update / gpu changes: keep the root-owned tool + its config in step with the settings
  if [[ "$OC_OFF" == 1 && -z "$OC_POWER$OC_CORE$OC_MEM$OC_LOCK_CORE$OC_FAN" ]]; then
    [[ -x "$OC_TOOL" ]] && { oc_run reset >/dev/null 2>&1 || true; }
    as_root rm -f "$OC_CONF"; return 0
  fi
  oc_configured || return 0
  oc_validate; oc_install_tool; oc_write_conf
}
oc_apply_now() { oc_configured && [[ -x "$OC_TOOL" ]] && oc_run apply --config "$OC_CONF" || true; }   # for the no-systemd start path
oc_reset_now() { [[ -x "$OC_TOOL" ]] && oc_run reset >/dev/null 2>&1 || true; }

cmd_oc() {
  local sub="${ARGS[0]:-}"
  [[ -z "$sub" ]] && { oc_configured || [[ "$OC_OFF" == 1 ]]; } && sub=set
  [[ -z "$sub" ]] && sub=show
  case "$sub" in
    show)
      load_conf
      step "GPU power / clocks / fans"
      if [[ -x "$OC_TOOL" ]]; then oc_run show ${GPU_DEVICES:+--devices "$GPU_DEVICES"}
      else command -v python3 >/dev/null 2>&1 || die "needs python3"; local tmp; tmp=$(mktemp); gpu_payload | base64 -d | tar -xzO tools/proxima-gpu-oc.py >"$tmp"; python3 "$tmp" show ${GPU_DEVICES:+--devices "$GPU_DEVICES"}; rm -f "$tmp"; fi
      if oc_configured; then kv "configured" "power=${OC_POWER:--} core=${OC_CORE:--} mem=${OC_MEM:--} lock-core=${OC_LOCK_CORE:--} fan=${OC_FAN:--}"
      else kv "configured" "${C_D}nothing: the cards run with the driver's settings (set with: $0 oc --power W --core MHz --fan PCT)${C_0}"; fi;;
    apply)
      load_conf; oc_configured || die "nothing configured: give --power / --core / --mem / --lock-core / --fan"
      oc_validate; oc_install_tool; oc_write_conf; oc_run apply --config "$OC_CONF";;
    reset|off)
      load_conf; OC_POWER=""; OC_CORE=""; OC_MEM=""; OC_LOCK_CORE=""; OC_FAN=""
      [[ -x "$OC_TOOL" ]] && oc_run reset || warn "the overclock tool is not installed: nothing was applied by this script"
      [[ -f "$CONF_FILE" ]] && { save_conf; write_unit 2>/dev/null || true; }
      as_root rm -f "$OC_CONF"; ok "overclock settings removed";;
    set)
      load_conf
      if [[ "$OC_OFF" == 1 ]] && ! oc_configured; then
        [[ -x "$OC_TOOL" ]] && oc_run reset || true
        as_root rm -f "$OC_CONF"; save_conf
        have_systemd && [[ -f "$UNIT_FILE" ]] && write_unit; ok "overclock settings removed, the cards are back to what they had"; return 0
      fi
      oc_validate
      step "GPU overclock / power / fan"
      warn "this changes power limit / clocks / fans of your GPU(s): values outside what the card can sustain make it unstable or hot. The driver clamps the power limit to the card's range; a reboot clears everything."
      oc_install_tool; oc_write_conf; save_conf
      if [[ -f "$LAUNCHER" ]]; then have_systemd && [[ -f "$UNIT_FILE" ]] && write_unit; fi
      oc_run apply --config "$OC_CONF" || warn "some settings were refused (see above); the others are applied"
      ok "saved: applied now and before every miner start; undone when the miner stops. Remove with: $0 oc --oc-off"
      note "to see the effect: $0 watch   (power draw, clocks and the hashrate per card)";;
    *) die "usage: $0 oc [show|apply|reset]   or   $0 oc --power W|N%|max --core MHz --mem MHz --lock-core MIN,MAX --fan PCT|auto   (--oc-off removes it)";;
  esac
}

write_unit() {
  oc_sync
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
$(oc_unit_lines)
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
  if oc_configured; then oc_reset_now; fi
}
fallback_start() {
  if [[ -n "$(svc_pid)" ]]; then ok "miner already running (pid $(svc_pid))"; return 0; fi
  [[ -x "$LAUNCHER" ]] || die "no launcher — run: $0 install"
  # plain log file: keep it from growing forever
  if [[ -f "$LOG_FILE" ]] && (( $(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0) > 52428800 )); then
    as_user tail -c 2000000 "$LOG_FILE" | as_user tee "$LOG_FILE.new" >/dev/null; as_user mv -f "$LOG_FILE.new" "$LOG_FILE"
  fi
  oc_apply_now
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
  as_root rm -f "$BIN_DIR/proxi" "$BIN_DIR/proxi.prev" "$BIN_DIR/proxima" "$BIN_DIR/proxima.prev" "$SEEKER_BIN" "$CPU_SEEKER_BIN" "$SEEKER_FATBIN" "${SEEKER_FATBIN%.fatbin}.l1.fatbin" "${SEEKER_FATBIN%.fatbin}.l2.fatbin" "${SEEKER_FATBIN%.fatbin}.l3.fatbin"
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

  install | status | watch [secs] | check | gpu [on|off|test] | bench [secs] | oc [show|reset] | logs | start | stop | restart
  set-node URL | node-reset | key | update | uninstall | help

  pool     : pool [show|join|rotate]   several VPS = ONE miner:  install --pool-server   |   install --pool-worker http://SERVER:8100 --pool-token T --key-file F
  options: --node-url URL  --own-node (run your own node here, mine only through it)  --key-file FILE  --generate-key
           --workers N (0 = GPU only)  --max-khs X  --no-public
           --gpu | --no-gpu  --gpu-devices 0,1  --gpu-intensity low|normal|high  --gpu-arch "86 89"|all
           optional OC: --power W|N%|max  --core MHz  --mem MHz  --lock-core MIN,MAX  --fan PCT|auto  --oc-off   (see: oc)
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
0ef3TTC1l47v2kHqHX2DTh8Gl303iX7el8Z0diM4G8Xd1mV3wpfRh3N2CpADBne0HY567NEb9gGO
U8cqHDf6jnbugrBK+OfEGBdt7bYMrCgRjXxwLmWRZ+bKCnKmDG+QB9h7Z0GV3TlxRbrGfUatSg+p
2PaVBNSVfxc51yRo6dzvTrdwHrrG0HAZ6p61Dx3en3Du2wyt0+fK5a/DICHTi+on1zSA+cvlhcvI
BLjytpfbmAelL8mxO1I/DkdXtFsNSjqqZtb2aRezmfUBVy7Il4KtprT9bL2jjKpcSLjfNgz0+YlX
+N69FiElGp6ukNwXTziNqkcU79G/n5CO/z0HMrmtNoITWtXzP/rtF18+/fKLVxtvgm4ngGr9LvDx
WX1oyiczF3j7nhHU8bvXHOz0/rSNe7soxc+mWq/SO144gY/XKqFd0+tanMyhH6qOY95WfrirfMVf
GMOBt0fDxWJka3kOqcteZudwwgDnXsSTKMLdpM6w7QdHDzxXrTBl3VriV+eU7raZt5JrpP3N0K8L
OVaaYDphUUuyIiENKkgNSM6p7Ey2TiouJ74OOWmVZWpyCDHif9Z7dridEd4xGLscMkibn6/SUfA8
rDAUgioDFmDvgn+4Z+kSJ6X2UMAR3vKYoaH6O1Sy4spaWK0RjvwOmyT7VNM4d+lKD/TrMVzNyqma
z08cwFqtExCuOD4u3tCQ8TzdOXBcyOByaGzD0gzmMb9W2F2z4vMQlEgk01P3/ZP7qzm1gjCt54V7
PvE3+Xmxel6uPy9b5cvN8qP156NW+dFm+fH683Gr/Hiz/GT9+aRVfrJZfrr+fNoqP90sP1t/PmuV
n22Wn68/n7fKzzfLL9afL1rlF5vlO/SpSS8U/8wg335GN9OCd0I49xHvnL1KDApNo8hPRTuVWWC+
RnvRMuEOtt0uHj3PMgt/WBPPO9LyIFibH6jpvMnKck1defV4nzB++LDtcTxbqpkaL6x3VTo1F/TW
lN62HpIvf7Slh+0f5b+gDSC+yjW14PHN9DR4+yn7C9EGiop2mfB0KPOM8glpRy6/BY7PoCcI5DyU
48n4aGzPeOGxs/fVk8HzwbMvv/p8pWVdYt7va9ayZskbW95CdfepubQiTRkicgrhW7oEnJKbcnQ5
xNCQeTOFvek2G2YZyCpuG63RtY9Na7mk9eA2FY5a9G/1J97dH9zqgfIlDNErjAptc7G8YL86bLdW
F+c7+hh/7E7G13ZyC4ffpjn4sjhdzQggf9pl1K2EOzGAAhb9LRAKXf6gChwbXFfBeg20IkeZY5vV
VO0fZ/1mk9szUZBgwvjKDxdQP/HrTHNREK3PPHFRGF9lLqsait9KEnrr8K2v6K2ThG8bo/lDQdva
JiTbLEN64/tO0rNvB4Gq72szkMhFLhOIBCcRgMR63/NXkg6uzydNdwkVGcqsWlk/DiEpjm2YqcCl
MQeytcEFqiAfA9FxZX+6/Xhp8Y4yh90DoDseQCOP7nPz67xeptRd7wVQlFjXTfK9RoXMf4By4L1G
eT9P3v67NswCt++CW5Q226bzD7sYb8Vr/wyLbeOk/318NPxn+Mj1xnlQq86At3bumUycnbHGcG3W
+D54QKYGFYmnf+9JVXOjrJnRjefCzi4gnynKq1cvOoJBPyUmrfmxv2JI3lyX99mauBpcebsZ08mB
cn+cec9ti0/45citz7nVv7nbFnltdbtP+yrZ9WCGVvDFYsLLJq308dXqJwWV1EuQnbgSXgh2q7y8
0R4v9fLqL+EtQlCf/zV48eSLJy+efP3oScAhOz6sxu9gCnxGkVBuV9mTFiLzBw7Mmk1J6hK9ewUw
hjb6b60N88qwK5yXD3mH0GePn3Eojn/dBe/VjhEO6+HMY3r05atnLgius+Y88+XCxqajBSjkRpW0
EV0aemnlt4Pftvr8i0hC3kZ+NlycX9jFUB+2AoN4ZyzXTyq5mtGGm82OhbyWyh6YZmDNkHe8C3i7
QgYbFLOFQXgO25JjcGh7XV4iqgM/hF9z944ZDMefHz770xOOdVwuuFvtxeqgR2uHwcPPvzwJXsTH
xy8iTCm/Ys1eJnvBhz9MAr/07jflai3uUrwNJ1T79e7VxrsQmdVwZAf+EI+5izbSy0n5vVleTPt+
a+nLmZpOqS0NP66OUqB9TRQd5VUP3qIOV6h4f5K5c2f1Zgy91GiIdlfkNT/rf/iasTjYsUBM8rsV
5kOyHTAC/555mDI77YAOzFncp2ADv343o9l/feHQG03ZnVJrhUKFrpW4CXpugXcY0HZ645q6idJ+
2K7E6gKaFvtkfFoN/udatJ12t2/QJpV+bgUIjhuFzoEaKphNLl0sL++Uw9tTg6lQSBNSxvKyDjT7
JcOkXhXR2jG1f+L9PZgDdVgJsSOHdXz59csvHz9phTI0jrzV7GNvMhfNmwO6b4ftQI0LNZufo3ya
3zL0HkBqQt2g/bP5/Idxw4pqtRbtkUKzYr9jmaeGK6shQRmqffCAOj3YMrTjrRbBaiG85jnfNtwb
7+a7G4tz3DTjlm31BXf6QQ/O2ly1Vh0M6x0eYXz58bR2Tf/otpvzBhAe/tHV71DnrjDhE4+ZwTiv
Q6iTPv5T+m0dxpWvuY07y4CtNtaxybwPhFvp2+agoU179+Cg2f4quUTd2/ztsyBy31prXnRQymr9
ahfU3G6DcSvuNuu7/PPBg/Y6wqm7+Mn9tv/XA7Ouz21ZLwT/1LTpWvfbsu1/u9nFuuysJVM94IAf
OPp7WFVeVNCWmz5A0l24i4YRhuPDlngZbHW8zA+8HSX3gZrf9y29PjpjjYS+DHYZ/NBfp11rVdyB
2x/o0v/wuP18q4CJ9TXZf8ruuXHR44NXOKg9tLbxD9/pzz67YY3DveAWOP6xolRnoYMXXv3kp2aQ
JpvzCTK9rdP9ko9vakuIy1pgs4Gm9XWr9DVp6TFef3Hzvxb19DK//fNakyzU16h3g/fHNMLSNL4f
X3KrV6aNFXzxU0LgV2sdvra7TagVbT7Tu+T5Tfall87q77zuTHSO+htRFi3swIqA18FXlf/oz8iB
kqet0OnwGund2FAudO3HfvN73OMHcFHWFwlG0cUf6eXOxfDQF9nvlrcINx4T7mrnACz594SjHdze
693aE3p2ewH8JiyoXa8Kr+bqdq+/Kre8Kt1jcv1V6XpXvxrvqFWs1bqlwcnuBic3NJiijXe1GPd2
Ndl3dneTkxuaLLe1OdzCHcxwNSM1HEsea7mVWx22vS2v4r9o7/zaYUTR7qMjbbhemmvLbTjYj0yH
g7fwkrhuUFsDIzdf3cbBkXtMrr8aNc3eycG+BnED8ye7G5zc0OCtHBzV1e5s8m4O9nUkNzRZJtdw
/yYHe/7fX+yQO7FrFalyZp1q8Zs5HwavOFqluVun6fBDf53SYxf452/o1itpXGRLG1njjTrbsYey
P5nW1gXw/vT42xXi599/5U2oV7//tvbbb0jtFXgzYc9cCii3rLdq4ye6nrh1zW7iGnnaRDAYNJwz
/2tau8Uafcx94xahmc1s5oUW3LzYcrN+k2jBTcc/7RHUx6AQXUUvjKtvtYh8EvhGu4UwDkBapVaM
/XlpfPRWnb04b5jGLfb4NvNHexHIN5c/uteppfRvw1BkRU4J3FKq1dmYz6IgbxEJyvraZzwEbYfS
dEcgFS+tUgRVN37qpHED0PKrX1mm/nIEVWszq2aZ+ZiCU17Rwi7vLe432PdHZFJex2Bx0iQNbkvQ
qWvkGLRVZBqtFI8uafl3lXNSb6fvthft7GneizMfa1eXbrvHXc3VOz7qxCuU568a9lwluPEsmHUY
c/Wj4dwfnC1OBD90/n3XzYaHry7onLIp/nl4GHx+GDw6DB4fBk/AzIfBH/zuvg3D8sPb2ZWLWL9F
3Ipi+bWaD4LfBz8QBwUn9DntPPu5b0r32al/9qLz7CM/A36g2dC+8djPmh8cQ7aa/4R7+LDb8Ket
i2wMNPz60+rNL5gqj/qtZdg/NJfqMyVX9zqPuxJaj6+aOiPBRbRut39GRCTSN02qh6x54BW/9bT7
1t/cmHXmoJzeiuVbmVLuREA+WLE+vqe5t9q8fx682sadnG55DWNuY8CG7VY8BwbAiHy+QkIPu8Lz
B2IU8FrrwiMvJVdj8IgLfdQd7DkX5F/2r+LavPvUw8/dbPBXNzxLDU/Fjn9+pNa2JeJTHpwaOlH2
/mMIvs+d7/Ixfh49DCgyLTjqPbz7eX+ltF2RTzwB2kXi2pNVkb1v7/61/3cykh5ywa0CHK85lg62
/6GAxxDOn6Mhn3MRa/W7YXjUrv6LDSb9SJy7pvPBUVv4iQMBanbDLTrMYba0nQgFEntnbLw2uGGt
hE0Ll5iMXAGN3ewF2t9WzsO1rRpoFKenkFT1icTDuVtJ8lv916iohhm+VS7HuteZHHUmwJzOGV7J
Z4hEaNflYfCOj52Byj+kzKV37wkA4hMQ8l3mDx52gvM+FeF8YlnXc3R12WxSfo2PwS+fXl22wtXm
r2Nap/QHlawuNDGgHNm+fkvyLZFuuxfxPRmvzgpphXpSry/diSF1EC4vJFD/+kRjnyvtONdLVJqg
TKHGFHLIzAGzlrAggl61pnFv6YjMm463JskVz7AVpiZufeceNS2JQdferb3Np15zAXc7oLzH47WC
8vzTX3Jvvft71Hk+owdaL9DvrH6Bn886DVw4ZsnqBjb+kgXH8KwefO8ejNauvXehPh1psTo0df4D
rNllcA/1d1pJ3Pi+1Ur67f5pZMFvmlnFN5arLYZtMF6e1v798ZLutXbg7742pvf8zK2PGkCzLsQK
iuPH4WoLrBV/rTqIJ/rtY51Xjqz3fTpdwbGZb9B77pkHyBSTzhrEd9Zf+Csz1DZ7gyplnPK+5jnf
eNHW05QM8pzn5Fve02Tc3o+FQgqV+V5piLEjPsLP7c4S9CYGzP0P8w+Kjcs4q4mP50PrhxWh/YU7
X4fTHmgFGPcNKXJ+3x3DTtmYQ9pVYj50iyBbEq9nqOvhqzoUgXbxQUv8yrxfPvNr67Wo9eCX1+D5
UEBavmtwBMF7ulAbIMAovYfczboZ3ZbWUNq11qGQc0V7jnOEBYcPuOgFDqaoz/vhlh/3Az7Knd2j
fOQ0mknC3gP8Bsb4jOBa8bi7tFpK+2dv6qDnhysPNKd5uW222R9NKWbuhHNSgT7VLPRKq9FGs8YX
6jDhyqFt3EmEfIQ8vw4htJYY5s8q5CcJKENp8XdiBI7a4V91pJh/7gE50E+CI+HOTmuNZH1OWYc/
eH6vbAk8/zrmZtDBH5iK/31IjOIutOxmeo7i+543M58byxRwTu7n/h1S7lSQe3JVk5YbhWp6TNbP
tawgNIKGgrwrDQjYbA9li/53LTGj565rraaJumn/vbWG/765BtmuIdmsQf7LNUTtGrLNGqJODbVq
9/wj22Px2SYz4s3+afsFcbraz51uPHAo4eioOcGGGLXmUor4qs/jm1O0jU/1oshj3lrfq+/25lP9
enMK2E7ukGU6j5snt4EImJ01B042UA+tZJ6jhF8+iXOlJtx5l2uEXT1PBDQ8A94wlSE55vU5Nk3I
mXGUua6M3pHp18UcdcohanEIM02JDbDIG4zR1KLwci/if+SlQTqy9PhvG75jmg3+9cGPwzo9khem
Ivlmq3XFXuVhgxxJ4XRVOyuuH4dthX/ljXZ/lbO5LnnJusGK7SWc95ctZdhcvbps8NQ1K1c/NUtx
Hkq2sj6vVonNzQMOWrZTQx0iBfXz/uazcsezIt3ycLTjYQ9HaSyZ0m75b/UgCFAvWBF6zRqTZdu4
XTtozTD9OPTk9EsFPw5XDr81JmDX7dQP1t48tzQx0YV6Tz+X/nmBKukETK8YaLOHo8XkyNkz0PrT
k+DGne9cFjX9pO0jAEvcUZN8qhQd3sSw5DuO9fvO7TARqIrO83FF9Vahgr+bbx6ozAd59uv0zcly
TF4N4j/vv3NQAnaY70IYHq9Ha3TODOvma3dNsunb9vg1T/pjNOqTw9w37o8bXc44d6nTYXTKP4X/
KXbNlkj66UJPy/qIDDRgLam59YpaZXDTS0xfFybizxk59W1pKI8PF2LppxjtP1tz1GS8PtwN+j9p
p9MTTWsuae2QR+MA0GgXpxBtBPQ6exc6M9kDNZ9f73c5JASn6r2CKLTQoJwH9Oq9tkm/9TywzVz7
1XjwAWGM2FaD0hCiIwbwZB3pbc688Gul9XPZUerfhApyWzfUdUPCE5PcOK703FrKO2ob8pm4Lsgg
SaF/iWQchZ31+22zYeU8AOxBWXXIck3m540zot40qo3Lg97zFXGBFfP+FvI2x50R1Z52QK8jYaOP
nzL48jqwc/npzssO36w69Js1J83T/loQAzfpw4b6ad1OBzWbg9xucZBc17j03zcPgatnC2+NedPO
mLwhZ70z5nGwGquAPf+1wJoNL+yRG5r5sjybTZZTmHj8+xmDpQPeKMCXx3khOeVeX0wMHqBr7Thv
DvjywdibO3sSF9Db9cvQA+y2qnCJIrF9pDZv3Mj7YY4mHAhd23DzWgTUZFlfKDHDOUUbK8wE+x5P
HHL5CobtdOp2iqXHXO7/S7zG4dKUxsUHGNXbk3JodH2iETuIVrKG9eWcVdTf+p0z6d2demEGwgjG
PTsvoLe2KgFXHfP7H9r8/nSbkZd3rLwttqJzYftH+bH1GNtOj1puyT90odxh+3yn7nYhDCkSljb0
taVl6Kdofm4FEYSW2FHrUR8XFq8UQUfsMeiIkubMws2e+LMOb0+2Lf3Ca384XcGh1qBsoeYGXvqD
x0tdunIHW+1mZwtP6DlkJLE4z97e5u4ufT7HWzg5SmmtrAe9BqPId05CGPLeARBoNInd4cLpNnnK
p3FdByi2uxBuhhs7uePDzkl92xwI5ooNu1B+7QBPdHLtyE6M3dP1EzN3StinG+dndo/LdDTYOCWz
LZR/3uJEo71f2bfErp3SSzge3wYbkvRwp7HTUhTwJp1wvuExK53no94yjPZGQAPr8nmdYEDl91r7
4HTGgE6i2xrJl221o3n3mfXzNNlxVJ/cieLu3g/S+HRFOB9K6zaK7eFRtkia5+/epZPOHzy4X5vx
QfdUZU+98elaCCP1dERaa6Nv7Y3N6PjRprs3W3z1oaYceOleHr7pt0Zw/aEHGw+tD7sDqy5GnNwO
m1sm0xD+e2//2/zNZ/qeBg4ZL45n849Ux/X7/wuB/6/v/x8J+e/9/3+V/f/vBQ+NmjYS0qOze/x5
b8x4+cLcuxiO7T13lrTbVvQeGIfM+XfiODyWR5SiKILewyl5zY7kcdhv70FKpS4mejIKPKsd3Lv3
G16ScFuDu1RJDynXX+m9Le+500LVaNBuwvGF6R9ySRPyGiq3fRtpgaevXj2vq+JqfJneHHJ3TniP
/aEauYvUw9nxwcFyTo8Ddp+c/PTY4hueGP5o6UhV//XnU/fQwpyc0M6qJyePly4SiLZjeg3APnxn
e513H9tyeUYffJRx/83BdFnWuYP/z6T8y3BmvbikG0PDG9oc8oXfvubW9Ezzbv1Y+5wD2OYLSlO6
6Z23J6ShbnqKBPBtS1wsRkAUIObNDcZ/b+3VbQuejyYLT4efW3RthgGPegq6Y26JiJ/9Tj2o9U6b
iMw23YpJ3XSv1OdPt95zfHMSfPI7RcN1i5bQSRfXteSWldzDvHzFZ0jPab93do68Q2XGZ9aeEDjl
8BxKoLXe8vl+UgLUuHPCaxuOfnFxzeZpPjf3Yjl3qULHW5l21ak/Y4CGetFi0u2DprS2JEhOONDn
uuHt8tfPBzc04MlsNpndWL2lpzqlUq//UluFPL+DuRqa2tU0J6HniOGkBMjnt3GZTGkN1nkjHd25
MA4vqLtJNgKM2hWl/VhRxMyw3u+GRkYF1XLGY9WiPJfH6Vm9c/v+MCCmuGr8YzPrIknGdBg5TQY7
Xl7UA+Gp8NC3o+d63HcEp2K5wW4La1gvlH1p6TCD0eRyjUtOYDdfWHL5nU0Wvl5eMuaiPscTdrxZ
fBwWJ3VBHFkG6cv2PZ/YAOJCHh3z0y8XgP6tF6V0L7pdhc3EumiZ+WT0zjYlqnl7vCxnvh+yHc8y
HILUFf7Cfm91p/8Y8pZcfcRS3hNLnbHEX87sDycnD+mHlwJqviYXFpO3FEb7zZQE+mfuzgMue3gx
HXVLpdoqAGl72XMFfULzd3sRvH1c5236G9mFaxvgc6txny+HI8yIkxMqun/Qjt46JpUDq5x2TB6D
AL1a+ZyccATN3Op5L+l3XwLxFTPC0ZS2laaY25E9Gy7ID2NHV8H5ZORz812hfHCFTz2/VMPF9hbQ
yt7W6qOwv6PRl7MhrMJt74iNd0oiQq/ZeiJoqOcodugHj/49Bo0vBjAIBugQrST2fnfvd/3jxWQw
Z+r3+n5UvDmE0eSokDH7kwZquTjvfUJnltPZ5z/UfPLC/rC08wWPXOdKawC5voDfPXY1/NTpxUtM
sB6KuP+ASj6m+JA7D1HdZDb8kWlw5zD4hKI61OI3vTuYcjPw+E+Ln+/0/Yyr/76maepKWd3o9Iam
2LN6jOd1ANqMSIZ5ddI4P+pYNRdi7bOjv2Np9x2EWlOWk3/f0fhDw3/nwzB4hZa2nfehJatpcGYX
A1RUE5Jfd2ou8GXwLyanU9mfeexz6Gfgg7WZgb5SOBuK63IGj3/3Eirvrej408/3PExFg0BhHh5i
lHUew3DOrnp3uK00EvylzTfbn6f+0OO+X50XTte6MJ/6PhyveA0d6x9zgnX/+EJNB9Bd0GZzPRuW
tv/7VQn09jHlHw++n9N0qQn2oPXeP+w/gqbjJEDpJVjrmBo/WeKjNo/4oSLlN3DaqB6uGqXUWIm/
O5D0yeslbPqUdtXuwpcVdEHv2+PqNdWucaV8dRBlhdswp0lj+aqhEE9O3JlcPb5E9C3tgBcJQeG6
Xe3n2Fm0al/dNEok2c5RrdHgn8xTx0SYrXzkiNVlpdZQOzHgZjikEI1W7xPqZn9NHnzztkfjwwKh
e6duIu3FuTHuLQy2c+w9ulgf/hY7tRoxrIJ3xw2U+anGFtA/Nax4d0wQob+Kb28e8cCgeaA7R37u
/AJ26znZySju5ASYYLGc9wAiDoNBn+mA5jRlM2RYE307y5DykLvbv4acUFIg4TaaNrDSU5SpaY8Z
R/aPl2PaMGUwmQ08zGxrohYhm5Y3eMTV2L+JKpYb/aQ183Gpv0O2d+ctGRkb8/aWU5NU4Q2zsrZh
/Kz8iHOKato5p7bNJjdOg38EEAObktNTDDQDrVZUrfU5DzjTwhHA99/NXduixjZmc6di7Zi7H5XR
GsqxW+MnasjPwU+uOkzwg00W67T/1QyGCXOMgyGLDixqSPZvV+j/vf7fpVEfz/t74/mvsYjDrv83
zLL03/7fX8n/6xyxvD+ZIR/MLHj4/MtD3p6Nzrhc0M5NfIJU0DMjdozQyVXDkrlmPjkWXVfvmJPH
6MLXf/7y8ZcP2UHrCyZPSe0nYFDh82QulD6nXXJ5CZ8M/zlhQdcmyuF8O1wcQ9lSBhufzDH50Y69
0xh2DO9ORrXw0YG0Nut3PPPHdsFY8Z1hS4f2waR1vrGaUULpEt+Ce8HzV9/S8RTDEZ+g5r3gR2fT
5ZFzDM/PvW8YPafSaN/Kk2fDcoZSWv7gqhqenPykB/qc4hf0YEh2qh4s/Sctmh8Gj6B+2k7k+dVY
A3jN9Kk71hMd91juvivitL7sz8tdvzzlNVo+6pOvPAWhR/TYpxfLha/21DlH3Dre8xdPvvjyW1og
hm7kQz3vByLH66tH6sOaVg/INGw/8OrpiycPH78cPH/yYvD5s28e/ZFdy1QQnU5S1/PN50+61QD4
pafON/fVkDQUsdN3L/mMzv+3vTfvbuNY8kT7b36KEjymCzIIgdRiXdCUjyxTttranijb3c3LgQpA
gSwTQMEogMslOZ/9xS8iMiuzFhDU9e2ZN886xyZZlZVrZOzL+2gekaQJ0zrXpdaMdoNlQIcaizri
BbLo5anw6LA+jaNL6BZU9Nc8eJ+a0Gsix1P4onmUaxhfjKFV8zTw7uCOZkc0ZIfY1uDpUcs+FxuE
KB+sEle7S4buQyxSjywaDuGCYYwcUZ8IMnsbBCHK+LCqT+6RQm22QClNmtlY1YKf7g8WY/g9048e
MVaf7Lj0oDATbeIo+1n/zqaC3rj04rTw90mU9WT13uMEydQKz87Lvc1OsWskMcLHoTC4vsnhz2kB
2z+9T+i9Ab4jq8ctwYqws5+qD1kNLXTVxqyzfZHOLqvOXDpxtXlAH8X1RMUnjojsQAtE0jK0FBT+
rrJbJ8fGIX92P8Rnr6aj1JmY1CMe6z1yphJNCgpMPv9BNwgTmnJA/2s6EDopnt8knvQm/XxyxlLg
meRmySyWkpazaHGCQk6K0BlhjyPUekNKAWk+cZLxzcbIGAoEdzJPp+ky4x5KW6Dno6t/jvbPZ4ku
n57G0aQ3mMfshLqcoixKICbBoPGiQYJRyIhO0J7Bt6JwVUwqC0apmMWf19E8HtCx1HRk+pCftX0w
7l/VQ8WXdGZZvOgNn/aiFZ8zWWgFy6cthZlgVYeD2WVvuEhPVnbpkJNWYPq/pe8TiKwETOnKTr2e
tcva3ecuoW0c0gxW9apT9Dqv7FUB/7lgZILYWTIlcG8p9t6aRBxuioE1L62yGPyEldpgRXbBK3GO
WJNaloNfiHupg3jXXrEY/0ajy8wRLEscgqyMDQ8HJAmzTl7bBVc3/ns6tuJ7sViYv11FAvTQqkA4
Y4zg6qm02yvhTrDHXV4i7eA4WiTjWKT7DsI7XWWAw8xEi3SS0M8RMniEFS/ewbGVmaiD+I8X2aK5
mwvvzMO8jpaENF5CoF55v3Lm6m4/zfdl8Cs+8WBl17Mx5UiqR3xhN1CGsKUhUsmiBjprQVsThgK4
2XFt9UW8rY/Vd0PatdbopydUZtVUPIbX9L26U6KM86S/XMS3LvIOHS/SRTTu0R1d1akimBW9wRub
DrI3WFwgprc3GkfHmTXp1eDbfENLp/us3O2cGIZkug4ZWjFPMz819NyZkqTD5VjipnvDCAzM7bO5
L6x9xd3w+sQJj+gKSyn127stdQ+IqqZVt5MTlzbV07uTRTpcTTy9td7aH+jnXSlnRW9jxnwqxnQt
InQOXNkp+rxmuIpe08FgOYumg8vVUOxeuhzF8p/l+SpUE8tgOzW8m74CFLCeUzWOtdutU1XkuwIG
mFSLGsBqK8Sxi2ktOhdPWRohy1UCnGBJ20Bm/HG27HZJ9B9+qqayjNlrKKy+25hEg3namxO0Z/eI
x50oKQj/B2hBfDGjM/4fjDnxe0FrfN8SWbRm8yW3JYJabd1RjYty21c3Yt4xuny1pC8Xo6d0m7Ps
Mtzk7g7bbf6JuqeAimD7qNlsfie0dtd6NeR0TPkDzqRA0x4nAxAxPVzYEp65xgT6rs6agFEJY/4a
D4wHBCqLiqpd+60ylYt2/Swe3DucuYrqo2prODdsuNqoBqveYeVwHpuHTi8FIwbADq78iGmmOavn
RbkN9UktMHz+DjAxBde4yYsuLEvWbI9bWQQZYArz2lWVWWlcY9LiufAceLvGFUYpz9/b/3dTepJb
onj17dkyOwlzW830xpgRWzXmrBtvi2RqgOj0FCYNaPnCaweMp6nqBc31Hct+oOQZn1aTBgRcy3R+
T5Np2NgNGk3PkImhmCTn3OFVoPeo2/0c+hz2G4PlD9zgvRDpF4uLg3jxEqP0znb+3mk0C9vXtuvr
Xf93zEKmUHBISE+L/kFsYMqug/tZAXwtBbjDpt2NHPD035lh3kQXzwdI5/E9l1V+H8/fcJDYPB3E
WZbOK7d0/eUQ1UEQHAEXTaFIfQo3hyjGj+9/6R3s7/+8/6H3/tX7/dev3u7vdUqaCrrXW0J/t6LF
VrTFem+oLAJcCo7r43IAHCHkq0EmSdaPT5DuxF9BMhLBKJ6edbtn0TxsVEyloYbBs+vgjJ2ZQo7n
aNAW5QbCUGIqK7CCBkb4aKl84c2pl75nXFJQuxT/FdQw9w3Zwpkf8LsX/ArHyuf4XauyH18J43Wz
j1d36cVoYMq9fOA36/UiGo9yHwc5n3VbRyWVjNfbG377w1Pe4TW6Kipjip3R+x8W6U/cneKmVT26
KphiVz/Ru+d4tVYnudKlqp8faY2CuxAqF89vmVqt38hNs0h5oxnii8qwKSI2sV73QD5aASbzip5h
WH/AsmDtfyUTpxW8wNtV39d+uXJQkaBrvnxLL1d+7YjKNV08Ny1q+3Ek46o+PuI1HaQemt9FjTjM
P+pbGgm3aricvH3gVuUxS6Kt340Qxhfysvx1WbL1P3/D71/T6x/obe33vhRb1QXt/UttUNFLLqwW
vo0nfOsqd9sTT0vf0aufFukPq74UQbTqS6CNyi8LQqf/rYigP/O7moNyRdLyQfmI1P/csiX+Y0J9
/oOyKOmPQwfBHi8ibZTHEQ0dmtfJAHOXMQoJ57SBX5phx9Ny4jlbH5F6UvFNw2VOiXunNs2Co7F8
YpyMu7lEG5yfwNK2np8WMRNzyWXmI0JlAGhozwP1piS/zLq+dA3PLavlnS7hkVoQerJjZ1uKnI1o
gotH08TWbPJwEpaKnGT3Zu0k68kQFSwGjOIqxdJcSA7MpT+VaJviVJWeo/xSgXaoj2Kp18Zyejql
LyQ6xIiBdXTHX/lJwm756tpZ6ByJ3kloagShOipsDaIZp1QTdNs1wWYFaYcdEb5rNnzY3On8LbgO
Hto+Fyf0NbGJsMTCE4LnALuClhB0DcWER49jeHezv7wkSPi0nCH8+lNpnJ0dGmdn+yn+v7NtRytO
E3NP0wBRARIGkGSF4WQAtnmwm8b2jnp+aA+Pdx5/3Qo+fPyP4HHn4kLfPf6m83VpSjoJJCZIR7xo
sbAUG37T2XHmq7OZR1MT4EJMuOzLMMmImb8MzrFpw/QYhfTScxg+s5jDIBCls7W1iOYEt1uTrDhQ
j4dpVKIJiMu5dIybe9OVDWAAC67mEJjpztxcAXxq3K8FRDLBBr7f9OBbtf0+q1Os4Fohz3tPwLPj
2F9wFwXP5GiMHxa5n2bItxPqB8ZhPvfTKIraaIvzqVaNcNYlAGWn3Z5WOEbi62HX9V7p+ELKOvPW
GQ9bKD1amHSj6ONstwm56PaCww6CigTf7cI95eju4zMD1wxZlxZlPeqd0RQhaeoPCWGLkwJPVzkx
nZQdqYD4zAjcO7SBq9FguX/witT/daQwcu3AF7sElMGquG9nleB1p+2yHKse3BlXFipvk+VbGzWO
6mcFbF190oQyMFnWSdzxcC1frDOl30oTNcxxaZIIkzErIAyz/QQEw9NySAKKVvDN4+BB8A3eaxL2
gAlGMkZepkn0O7V8AN83DcCx3NFyITo56xKSu4IkAGu1ZQFsWuL3gQmF3zwmkSuQX580v0MCIfh+
8INtetCeRBch8l5m7A9i/UBC7OSDINzm9MydprR48oiEsiq2gg6IZug4XsOM/g7FGmmRCXK1kYxw
li7neXQ3/G4Cw2dGTNG22DdMqu0ydnc89BLiVDilJF0QKTMb2cqOQX85JGBruybZX6N5EnkReuOo
j6FcH5lRT+bRNUotfUqn1o/9h1gQcWqcoEoizNktpuT9psYFqeRw8Ma1UIife1Yipez5pQxKM/f0
yQ017KnjbCqIo02ApDspgop1ccQGIf1JfynBXSx9EKU2PpFqGQGFF7uIZs4Vvx82j7A2C46eyLS3
4DSdERtMPknMcJ8LwtqMKhxfyx+ixIB3FvSNib+cJd3g+XzwbW4aYo8yguiu8XayBi7/ANDuTA5V
7Ql6xE4/0UCyFzvOUVhQFrM3p9Y1ni+2lrNgsWQHDXapG8XnEpXrna49QMNP+KcId9hRBPDLfb/4
wx592MsmPcTDiiOgMbQ93NG5DhF01VUDoD5hZ7LiQ7pU3hN4YNn+3tMfzyxg7CPPqsbelny1hPGS
+qnIsGxTJIg2r8U5t9TZFAH3WRAqlRYACr5WVxjxdEG+ax70a77ghH0RKcpaNC7i3syDoa2DIwHe
JygmP9noZssMi+OYFHaFPnVqwJXjTyPgj66FTvUv5pxgS5ParOhZ1jY+ZNgjk2C8az3KWo5DmQ9l
vPhucHhAP3cDqGV7B6/ffTw4MlBJGEQ9aewTlHe1p3RjRkYPOnL5uE/kCavVXRdE4xKWT+rGdde1
03G8cdWJVnGJWJ4832ZE+cYXANpWjjUYGQYZHW5yATyGU1Nn3YP95x9e/NR7+cvbFzTOYYiwv6OW
CKa7AScIOwz7DU7OxVMnGZtoJMnZgfe4138kbx70H1W8fGJePim/nM705XTWaBKXJtbJHI1gwZ9Y
Ask+Ea3ixbScNTcJCJJ51gri9nE7CBuvd6g3DhukFUhCYt0vm8NLnnKfhNkIqnkbFeF48a1AiaGP
yAoITLrJEDGpUektRljLp8+aRy6fTyuqY+8Jpmij0W9bCT0osGGed11NAJqz5p5kiD3WS1Tq6SpY
8p7LgI5w5VtI/vYIig7wNDmaYP3zwni6GSP7mM0mkn/KlcaozxW6Bwe1l9QO4KhdbjbXsvgambJ+
Udk2esDuej7jVlQzegxc3RgFzWMzpAemW0/tWCkoGVq1QloygMvg0gQUCuDUCE9C4tfeulVLK6pF
Dc/Lj3VCVvoA4BW8fWQTfOVppZQzSqYwpV+L3ptRiWuok6WskkFGd1pvvQqvQpGrix617Lo9octb
tnig3JRHq9PEeSa1UbNs9ahTVRm3hluM9hUCkAwlzCswAfZe8So/Y0NrzaiqOPT0GkXlUvDVFQPs
zVdGH2W7rtE7NZqF87kphTQGoYgsQor4ErgEqAYydKlCLuxaxXFGw5RNlryqQ7PKDAhKK0RcF7mm
Brnm9vPqw6OPLOilAmIpEseb6bYqhAbWS7A5HUpmzmpf07uqyqvnW+3bUX5icJOIlFZUMiJSrtmS
876Sw7kBoc5XoUffciUV+pUlyo6RKD15sc5jBEHoZkZJ1uMcOCXlsAOhZch04LEgWVbDpSqAa1Xk
zH8y3lJOy9WeqN9etf7EUbCbgKzOmjjZGIkUL81auWLO2IlKOJYE71mzTnPNfCdnAKdPxfsadr8u
FtRLR93uty7v+Sws+dYYqcT2YQO5kIO43FqUg+WmTx4V20I8gfNSPKZpgKPq4Um4GcGTbFO2wTOl
MPMX8OuEeSwm9AZsWlb067RqxbC3UjgLm9KyS2vJtFsyo5tmMZPKgQpILNxITUqVhIxoXRMcw65p
YUas0XQxJo6UpZ5oqiIayd2TJEMy5baxDOW7wMzlZs5dqulwc3g5DV5Ow9zjphIgmw59ZRGxwFmO
FI9F2aVzKJWWHQPMvqS0FiXmgkptz1lE4dqInNvNOjznurGADd2mEV/80jv4SGjzTe/tu7eCNV+9
/XGFkwubbsCsFm+xbk2FL25zFSNy0nWDAdbmRnT6EL9gFmrB0OGHk1RRHdq63FtD941Q7pR2DSYb
3bnr6+BkpTFtpU9QNWkoxnpIspUTZobYZ/5pC3ds2qynqsNa1FdYnXEjyZX6J0wAV5C/O66GKXd4
AkXu7frj6BSg7TmSQYivAwp02wNTgLXcFbsW9ywX+deGLNlI141K95GluD99F0WfYXQZ65MUdfsx
Ko8ckjy9B0265AhUPqKsw4bY0ziskjWy7VVvQ/Asg6E9MJ8m2WWKwghCgcFTRiWUEdhnqPjjKoHC
gXM1EGOZK4QGQ3dFpRRdqIAgREF9LtjJK3ggSbIwKNRmp17aF0NiHIshF1O98m1/aAe8gKx+YYmi
ffTjjkG7iuFqnB0udDV+aovdDT4hZCyepcR2fzLByaw6kVXQitIMrpIcRM1VQ/NEYlpJlr9mhkw8
LU1srKdloW10VqnXUTf9WeWCSW6bxyOTWmV2HcxIgh+Xlv//LOOllqJTVlFnDtJucqNB882xvZxI
nDWMdCBfZYFx1uBkCJYLbbFGVLZSlKiLVNkEO66zjcgfMzB7rx3zPVblvECeOqXK7KRABJeBkIJZ
52nwB1YyzBtgTqYKHn7aoZn3JfaYp5BMz5BncQJdLXXZjxfnMVfhcbz+HYA0riqahlWsTMp22b+J
2d10o+cNDyamCzlwDS9cJwMRe/lVHCz7krPvuNyVbDnjzEFF5hK1ODbvS+qhWeK/y8Z4SewPX+tD
/P+orLyZzWFHJABClKaLfOkFoAovoSDiq156i5h7asHLLoz+D4PeV2PNW/iySAUIuqLtUbvgdNoM
s3FbuVygrH+At1VOrGmkidwPtdZqTh30hIqb/qrbRXP4AB66OSakDNfhptlKH+VrhgXz0LBgm+6g
5mWv0KxgsjcbUSJZYdT2XNmaYus18sKh/nLUzuVZAVmuC7ddIaObFx1nO1u8+IIjQBX1bpbdXgv+
dGEuuxYdUYoHVDp930+Yz5+pbWmLc8CohgrXobixeljX8ZpHBNdR6NHxwPZ68+zGYbOEoiVQQEzA
ZX1z12QoZQMmm4A8PXWT7swiGbvIVTE8KgazTAHLqGb8dfGepbwVSM/FXD7n8q/BYavxlMFtJUWC
czYWFowyPNfOF/0UKyK9seFOoLdCVLMgIevGIYCu7MVUjeBrp56/KOrdxXejVvleBJ8P1GfwSa7z
p7Vs9SYbC+sQDHVrBZ9mbc518slm72A7maR6BrBkbT+YHmBhYGclTbwLNPHq1TAGSq4bxXPlQoDS
b9W5uF+ZMP+7kO67Av0qwsu0oFNDVTtKVDss71dZdI2KL+dc/evGt6XjD8nM254naHS7Fdn1orI+
ru04XysAKubc5F5z20CvZB5prUPhHc2eenJXWpEcImznUG5VT4Jnzuz+VOpbSXndDfzXkd8qArs+
JaZ937g7HfbshRXgUnB+b4YrUa7vTPnPwCfIvor96Kzu4FqBC8J3Bk8wBUWa4fh7WcxbW0lLKgGb
eF/RvO7a1OKY2xYv0cOnbF64I/IyXjSbyMB0VPSpNWmZnt0VgaGI3nxxT9IQZxpF/e1entis4C0/
lVxeeVuxhtwd5RgltXZWa5Jt+cNBO387gglZ62bHYFfZHt0j+ZX14lCXlOZioQkN1sNG4jbihwit
iZq8jzbdKdeDe/kbXthdPhBrxu0fFKRHZTfo+NvD5Kw3iJNxWEJmTXW7/BegU7XJ/Z+ETcUg/c8j
U+6nFpfOI9RP40QAHb7mxQtxdAeESp15q69Dp2w5QuOaa1eFONF8cLKcnmbhU1VWDa6DHDd1u4v5
pdaENFG/0KMO0vEYhR6ajpvtF4eD0XG4QHWCow0U8cSvxn+EE0suZ6gfcX83f2KR/1U6GoHPTiHp
CCFQnavJ/iYNskIWRhKovPv6wM+ih9jnYqJGrdeC2R0Ze1dlukbPjCFYtxf/cS+0U70XViB8zsJI
IPrkUXP3Tl9Lqkb68pudO34J1WIreNq5+2fQDuHTp3f8NE/bSF//7a7zPcVHT+74kUkAifCGzl2X
yqITf3nXUzmXNW537rxFWOX29s5n7Sw+fXTnqU6jEX34cKdmqkUGS757Bnz6eO0zNEIXwzpA547f
aVGE7Sd3/M5IvSsAvZaBdAD8r7Ti/9/I/30aX/4r03/fkv+78+jhw+1i/u8nDx/+lf/7v6n+44t0
hrxHYN6i6fFnFYL0Cz8SPGWLdB4Hmig7L/d4jtrki4CrALIAiMge23wEd+JPPOAn1u+2JH+mdV5/
L3Pi7vJBNIEQdHPLKdjfeGhCZKV8uu1yKjy6BrXYKmOSPFwDFzTMIDtB6jAYC2kgmRla4AGqts1O
5hHMnYjyGi7nsSYH103rmckRr/Wz/toKft7/z97H/3y/39v/Yefx4+2/uSnAoenqdt/DgcjL0/38
4OD9Tx+eH9BHb3+VIHFidhum4i26zNs0dvPCgQ93tuDBgeXYvOfeRoP6pYTkoT038rizMhiobTJg
Xn6SBfF0ML/kgjGSyd39IEuX8wG3ykxEluYRA6AMQ15hsIklutK5TRRdEMwZTCBO/Wy3kvviOEnM
hzv0K+HEXh2m72xp4dOsTUvvcVrRe3ulc3AYzypvWJIVRFVPywYscDdXNwAxTlTl9N6SGBQNQA5t
NZMbuyRnw/Z0anB8NPsKLthp06NJhJsQ53ix3+W1eEzWN3YuDyz3LlkGNDQGuzaETmWcDHjm0XGU
TDMphzibJ2cI4ObnWXAej8f5JPXdHqYHXx19ALAON/PprbH71pEs5sxqJKHJce+aypfDNizocqwZ
TTSmAWS0w3abGuoOkviExk23NmbhcmfsnqKwKtcffhSA72EyJwEqJSQBT+GhVjYzgUUofkd/vvqh
JcFF+DKeniUkhLKxnIVsduHi1xEB/lY8OEn1ArVRbKZwZlCjb+Y3vwb2TYiID/mu7zHSJ+AQ2Cjj
joEL0CwDLnuFujCnnfFzY4G2iah87NL000xIuoaVLrnOaK6nSyF/gkA5/vA6LIKwgWx774IQ3yAb
XFNu2QmnS1NYn2M3oGMk3MnH0DMP8pu7jyRE7pUDnvvq6uYric3vBo3yhd1wE4+VszAOoikqVmqc
pIsxOU/eLoccXvkbeyO5wtx5AEaLtT5JhJ9PEO+DnAE2IXUfZChH4gXYbEnwvQSPcCgme96gN0O2
LCpGJ3LPTuKLMCtnHFmFipHVj8utDWMut5ZpkrKaUmQ84USqe+IzvxSZHclVefShdxEM4PbaK/YK
cxBhX3ZPbEG1zEW3qDdEeok6hvfvL9Z+bf5/QmThf1v9n+2HnUdPHpbq/3zz+C/+/7+r/g/CyqUk
rzK/QCpCVwMw/swlf7qlEDt/pFlkkPZlB2lfHgQP5ccj+YFcMCwNMMt9AO7f1nkHS2nkAbrl0+EY
pFurAc/jESFKTPGTOzJNSsu8z8VFX0RZpP3lEQYQbaDGQC1kJ2gegYqm+BCtvlssN6MpGmTsXz+8
hI1thpAxWuMkGY+TFN+PeAw1Scy4WjGx7Qh4PohG8YLQMLLFiZck9pgQoilb47JiL+iVsz4Scs6w
sn48Aq+eLHiUJOOg+5Z2d5JwCP883jI96mrQmf0STaTOdV6lWuoqUyfJKOGiBTIXEmCgRpbd3JXf
aU/kN9rVXRFw5D2JNVKLt2WKQqosg0/oJacFNtHSeXSsRL16ajRT6KXlxDi3nPIwFSYMO9T8LGb+
vTeMxvFpt5sNonGEgn38c9cTyM7mIxKveGflxZz2o9v9MD1WIewk2qGJ/5AQIYWXw0n0eHunWkD7
fjkq1W4yVQ2unvMv36P+eSC//wJHSP0VdRn1V7FjmioI7kiq7geIdbu/xoMfUHO4NOLV8/kAVSSm
wzMk3n9D0HrRCj6cvyYJ1+1NPF3cB8kEYqkpwdwKXhEcstfLAbGr8eRjgqi8X97S/u+/f/fiJ+rM
xofv/+CLoW/fvX2xb1KZ4nXDFH56/ep7ZEXymzt5T/W9bf/y+cfvX72tbS6vqTWYoryONVxxug5U
SwV268WbJxxAWVzRbmSaz0KTRqEHzlsRTGL6WjNCJHDHn0ut58yUsnr3+nXvt+evPvbeHHC5HJri
34hamRV8oN368LG3/+v+h//sBmZ7qVFlseum/ez5x/0eMsCu+2Un/3T/44f/7L159fYuw8k3KMh1
yzcPMZDZbtS8jsazk6gnKnM36HsRnRIeCfk1vHykwdfBU6UpEn6/4RUee/3qzauPbukxZDMwdVDS
6SgxtTn5/q5VoB2vIIeD5+0GelHlcblGkibgsr1wkD4m80x7GkWLfpKPot3pS81UXT0HSStmil1r
lXn1rCrlA3lgYu1N1hENH9vjpIBurowsukTNbSToiC+iwYIgFg1HS74GZ7FxZZND+SrLgzObzhQ0
Nq5mJnqgalSGNzmwBugUZ3cZoQqgphmZIBDEJFhovN5GqodW0G63ZTTrr1G1Q6xVQl5lB4hs0LyS
MRJjuPagTdsSqnsMNZAhFsspir8D05qyE/T3G1Rp9YpsEaVZJNF4/4/m0UY8XU64iULXG8Ia8vXr
JNMcHC9AT+XX74kgn4QoWc5dKnz+O+ESldSHzhmry6j5070tXZMEQqHUKcGFvXg1mSw5Roy5FKOL
g/lNS64rzRfsZg5aMB4nsyOYONtud9o70H8A1Y0jGINZA3KSApUh2UVTfGykrpsQim/1eEx9bwP9
QzpvxJt0LXHwrMMydBYlQ1Tz61oMS9wGCKxsB2INaH8cgugFokzZh3cq/Fg8VFzBa6NuqDfiovbd
+BLkKrGc1GAcJZNM3HmB1QMWo1lPzLciXiCMYQh7fWqHnUr0hA2bCFWJiYeAt6zJIxge0eqnjdeS
TSPEnBd9DC05tcdkBc/QmnpaGy6n+A7oHECpbCox0WfFc0DkJ4HXs2fOTbEeqJNoGJsKmsElFxxW
reTC1WUz8Nmac840rJP1coJaVqCN8Zna6h36SPv+2uyQKBPQ4CwaL/nCXgbn2GoNQWFr/65zkTWE
KDKFtOyoGlMkcTDwLCJOt08z+SQpgxBQFAjGS+YCHLoIoeTJggX/CcEXWF3Zbw0vyfksk5iHFs4M
EeN1Ezvk7CnTdygHcXfYuVwASPzH5Q7R0gl9ZJzfE2sC4hunqQZdo4euZb/sox6wrBk8bEqSJike
4h79aBqI1gulqN0XHI2mD1wAES0ZSEOzVXW68r7TbJX2RN5sS3ya2Rd5mGcnabb8BenjVmlN8oLd
fxxlH7b0BejB85cf9z9YgDqPIKDMxtGAs48dcMwY6FqsKUZN9JlBeUg3yh+WIsToroI1Cybq7cyH
mNMltFQ0Ef6v7ccdtBOoZEiyN36WDE4D5ADjN+f5zYUrOwGjcWX3fQPHxgkXO9jGboS5e47vjxSb
pnwMbYuHwu2c17cVz9roYYbEjtyikCNmACX2mP3QS/rXQZuD8Wp0r1LerUeLM+6Tv+fleQyGcbsU
v0dsBof+Oquj5fxeCD7gfSosu3ds1m0Bpn6fbLs2IbJkdIlY59BTWnP4vaQYyiM6HBQVhNBPTKKL
T6xUJe6yi9xhfKtVgzFIQQGRmvf9L0hARo+nn0xKXolDZB3u9/sv333YZ1OAGdmALl12ltRNBjw8
ZxWrwQx5EB2rOoV2jOMZh62ky+MTC1scX4Lp2fiS2FRNJbR+kfPiRdC7y76izkQOelzApwRyMH9h
7Ep9fl1OjJ47iTavBexbulyExzz/4mzYQ22ELG3RcSxYbvMropyLZADHKx28wa+7wfFsqdqcYGtL
lE6/fHgdHG5tMacffDzC78Tab7Ha/OWr1/v8RHn4oNPa5r85r+rb52/2j/4+/bss4O8XO51SDPSh
k/o3eMufKgv+NrgOzB9bRNm3sok2EHFAxr61d8M/v965fr19/bpzSMzx9YP+k+sH09mRTDXdAvta
1ZW3G2PiSqszwfCbha3brVsBShpfJIvb+mWdUcWuyuXcgreew2zV66rWHa8PRjo4PNh/8e7tDwdH
XsL6KFvORRQgAn8yh93RY/VKYzjmj65b1eWH/V9fvdg/aLnPPj7/8OM+JHbvKauRqh6xkungjffm
1+cfXj1/+9Hv9hcUj2nV1JQxoopJ+BnXQoykZ4v6xNU0vd5E3+E9Uo1JKygpXaAvq3QH0GsI9b4l
IbjQg9GxzWs+i+ZZDBfpzKMy707DAVeFGrQ2yqWifOwRzwjHLMZT5A2Kb/4+Zau43v1C+gJRpUlq
3G4XxxrulKlYbjqWuh8y0+ezRHwAQpp/29T2irIe4ThE2BWmH/FMid4JzxI177SQKWRflslUpc1a
YrZi3bqihytXNB7bFSHmzaQCL8x+yNMb/u+fM1EVbDeka2Q7g/CMuveZW7wNYM4uN8jfU5ifAxs3
f1/gv2zSu7qRP4KDNxn/fPM9gczQJDzEr5IcbNgeDFBnln9u42cGrfGwLemKK9M4VayOFkEz883P
D3OTc8cmq8qN13Qumm3W5GOXBEeSfFu4sjahTvxyPbzmTdKz9BISIOgTGR9awfX4mvg5hK8CnYab
drlN18fb7jomUG2Bv/XItSKBKT2vjtbBVfc73Exnps3djTVgIt8UJYMGftkhR4zP4SZ1W4DhkX8D
R827opJ/9q6p04kzWfZEqphqxnPJ7jRBdQVAvyoAX65727ZXzXrGVgqUOzI7J3YLq5yVNCDx0CBX
juiBP7I6UDO7NiCeroeH2nZ3o/46s8qrospAPxr6qXzyiw7orKxzyMmsiGb32MJ0qYm9hib0sxVs
ChDRL7JQ+sWde2UxxB6fRL714AsIa4RXN0hrzuiE3Q5ZlRgdz2OCdus06LEs66CZipj9OnAoY7l8
ajSjl8+JrfrBVOosDVxXu5G2/eu9YPu2DD03ayM/PkjOCgRcZ9DeowLaU18hgQnRfcISAHiyMPN/
MJAQx3nHE3KBR4hRq6iP1mKU65MmmkU7z2koZCKv7MdpFdnRhas7GLTf7bKmqvfsGbWWkpetoJRD
08tzKv55zFiHm8cwrFZtFSTNQSY43xom5E8TKJb/rsYBeWQSIDv81e7nbWy33bkJfnqQrbeNWJhT
8bDT7tQNjASsaIw8aAw2KxJn1l+pdTNn/l+PAVzywFL4rcyHK62zL8IfywSaS07fEQ2Hc5pB8FXB
jeQrgsg5tCe/p31Ye5q18sIKWSHnSMQfwSWWL9RDAb9v2sXoFYBCwWAZl3xmonN1uhEtrNGPasv0
ONy03mh06Yy2CT6F7PYeAPbp8AkGQridqFl6IVgEUxCoZ1c6urtwpZNr2xY/XXaxhT6W2T/2Zmvl
R9I0ZFwkCeSscKbseBbIg7wPswKLmwuomVkIRsIG8cqOtAKDVAbq7yGpdvc0UbbB1/JV/rd8nf+t
a7R/q9NO/rfujD2bQryyg7sM3urPkPwTe8DTuRuG4+9qBqPmSNVozNC+CdpqxtUKLalFS3Zn28w1
PduH1vosXxtbs/xljcvyZ9XKfVMub4Q1uLq4226v2Ep5i5boNLeVdruwgbop98RhpNvNZtH5NITB
BZkRy+lBFxGUCO5tILi/WgfLGc4BF2hvNUfwmZwA4+nju7Nx/gW/oiXe0Dm8+xj8cmCQ+ApqJEAl
Ov9s2a/U+QMJb68gVJgBpDoQkXE8YiM8kAizsy1WhFF3jZpJ1KDNR8116V6VMni9NORVO7cOQ76r
5lZeVWGeAFZJcSEMDm5mHZ+zSUMWPocKUox8Id5SG+nGoLa8I4PcNgU7ex1Xo57mGsuHTXNmow3m
8QypOodSKC4rLfafBKCKY39cn4Hbo6MFMmC2p4Txi0j9Dki8GqmoAZuOxx6JdxJ++AyMsJVtqwjo
ruo+4+lZb7qchFqGIEO6ZFXPsrMGGyfgzHW1YXfRxkfwHDiNaRuoCdEfXoXuNitPQ2ni1unWEZo6
CU/H6mRvZgpT5W3PCQdYSes5Q5UcogxFyffZoUr2YQV1Yo/7hjpHInzJLZYqe+Vi9i3hc7YYvfuQ
lgy9JHc+actlQiVx/v7mTn+6yxCUvDk6PVu66PfhOBpqJ0Ypl10H91wNmrtNuY+WAZBGldWAVrv9
uENi2jiazMLHrWCn0+k4/Ri/rvr67mJXaNwORSQCwhnLApNZxeb0OpgGz4LysJYBuG10tWr8KZMQ
WRZ/6qZsc/IFZ3KWE6mflppUGs6pe3Nx48jsHKjBvTNXJtncDM5gz2xEy0Xqwq9wO5UH+8vb/YZN
QJ5/UeSHWq5ODiWHCkJMsLVl/M8Zo8E3W+gb4s8uCMkP0skElit4UdlgU/VDmPMM+Rs1FJtxCtE9
u+huTHyWcYpzOqXtTeD6l2jFnTzQ66wc6NUom8kaJS8H9VJjL0gkVGlnszFRka9aX5lD6OUJdisR
YEGJbQLJxvWBZI4uGvmwEPI2XhFRZpIMeQsUvNrOTpOZVaxKZSIbRhdxCQ9ktoE/SNisUGOJg9Ne
cI0NF0px7X6i+RtzGo92N1rXNpLPPZquJh6r7Cms3EjPDTayGQEP+46erHDdaH7XKn7HSN5+J0Zy
3Tz7sbRB3cfi18bkazsw5IH68OmD7cx+UtUfqIXtS2ub2i+nUgS19JHQA/uZtStIJu7qWeg3NIfy
JBSGGtWsPc74zJ2Wab4qMfu4pn6Tq+acQZSuuCh8Se7ZW+LQoDrBQwqyzErYmG9jWBNll/tAXM1u
bIxdc53M7rdcvPJHDcdrIoc9Q0fdzc2bNb/LF/Lk0Ypl2E/sImpJrs5FSKCdiE2+5d8CbeXOQ8hb
zTwMPTeTkGRddaOri0hhEkqWa+ZivrnjlPSz4vaUiK/OT2mwnZnxSCnOybSrvNfio5Kf9ZL90kfR
OItbxRq45WAxccxCmW1NtcqxGHjX1qAx8+IZkn9GbrVzzfrLhXMhAGfxogSL8vGKC9/zYFKbFy98
CcbhV2OXLFYyx+Zd2iR2pKlu7/iRu5ADhf2KWUNdn/sHVFKsUuWCM9ZonK1518QXR11xLEC1asuR
oe+n5dcVxbVKW5DbkBRed1Atu6TQLx4C7c819uokHs9qtsr1Kqh1NqkRizsrh5dKBTRodSoIrtWL
g1lyaoArbo4dbFUyLXCjGR3bcO6fndJVmfimEngjtbCM2w0+5X5TShXZa7FYJLkThHFe9jWZcnGI
FpelZZd0rqjaXy4kg0eaIjGndbOFMxfc0RH7M0DaFpLd3XHH23Zoutnt8Y7/50P7Z6SOWxq1yWtj
LniSziW1AuYhbmljGAOwDOvXLwW8tA6nlIGdDhOU6drNQ0CG8SAZxpnGQVAzrrybLdSDHEjpU5Mz
MLjOCLTnxMKJ7FxMBVoqzPmsMt5dMnTIPdx0eBTHT9skaHiWZzbOocBcnZxP1azWvfgiDldmzmhj
D5kl46LzLIH7MGGknRs/yYZklS0yLXQMr5F+uQwwQUhgwbFdml2yFbzebrdfPwR/PUGqnrl16+dz
NQUU9Vz5DJDocUDn3T7RABOkLZheBtmyn3FxnUsUjJ7N44wlGr8Ws8QTwS1bPHPN6XIQh4Ir4KBd
kHEkpsyIAaNM0paHm5wO0DkjWz09bLzuNNxtNmVhSxYlZWHRUzs/A1fwtb30xmmWXYLBg08QIQYk
cHH0QL1CSl83swdCLbEAVpihu1lPiu2FDXvIBRer8dkY/OZh4/U2B11J6NXDxlGFdn5mllCApVx+
wQRu2lc3ZrQW+sfaOKRoEGWOiaxQFhEHULH5syoqlZ+Adu8cQPOWEoHWLQGdVMqQVUjaTdERTcWJ
1KsZGF5xRg4EDxGue0AIjv57GGRJH0EeBFahWzuwa2KIt6AIY4NeOzsJpKAgMpPwNtek/DHpkp2s
NTadFER3jpcqJZxhvyXml/opdApze39aiIfX5Da5G67ihg0v31IdDqzL8qGmXyMNtolWZousarfh
6NT13ansZzXJZ7KyTsJEE9cN4Kcr2cy8nr0Dd5KdiKmVCc6VGeDGJCIhdKRGWLtIN/GLntFvyXBx
svUoePv8ZTBMjgEl6Sh4egElUfC6FVwQOjO11sXNKpDA8yD88PJF8LTzcAdZSaJpBkSpwXrsgkU3
LAaBzpLjaTJKCE6FPo+SeWYqADm5BuAOTZTvaXA/CC/ofz81d4OfaBvxYJYitw7YZUmC9Fp1TBHH
poXh04umzBepaX+ysf/5VeCBRXlB20fUCxUJiT6PIuRJCobpUi5DEHJmhBzJM74fJdMkOxG667qa
YZlS7JdBjMEuPEzoLxPnf9QyVTVyWnsCuYQD7rvdIQfgh+qy5snjaTl/1DgtZ4868fJGjdPDzlGw
uRfsPHpqnzzc5kfbO994j673SJradXVxzNjM46EyS8Z4pR560Yzkfg0o54uaMasyoVuWzMZE9i7z
TgRERDdFO/Uo3AwPNF0BT56JUY+OrMcHGo5TnJzbJHy6fNpk3ahU5itD7DSdbkXD36MBcAhuB6Av
CsbJYjGOTVp2hdV+TGg+2PmfO48fdg2Yc8I7xEzCn/XbveCbloYbChd5nhLAJMcn/XQpGTZk0RgV
lREIgD9oES0OW9Y+BfKSuUhzDDGyAad3AxQEonsLefJoq58QWCSTfhbMxsssmKfpxOUaNHRxGh9H
XN5Dp0T3LiByO+M0YR6IXQiEIR3z46PcYSIBxu20248cTHVxmCCx+RKhd3yAY1Mx8fQwQTrodpt/
IA7+qI0kR1J4Nk/rvFuh2xQG7rDjbUThGrCyr5M/JLqo5S6uLzT9PaZ/HVwY9zPEV11vwvuYXYA8
Fek9/TzcvKjSirJrX2JuheL0C75SqG5ZNkXig9A0eMzps73PjXNz8AyIrMwuDIMt6vTJKgmNIOEC
rYaVnXZWVPLsp/N5imThQ07LX5yX5bNw3rJ7kuy8TmkHa+kZ+BjqctwGThjRrQJjCOutjFZjer8P
9eJZjfeXmWa/dpq6YtNyddXoPrFCp5/n+VZbrNxaKfma0ZlvDZt/6qYOypuKKEke7zP21MxzcNuW
asN/3Y5WuOzRtT+ke31kIDN56nokTQuugAz/zwjUtz3pwGCpx4V5K6YK+eezZzB+XWOhCaGm7eDb
gNqjDf46Cr79NnjiB0ZUG2NCzn8PDxw1cXO1j54lTucWEz0V5G4yDBSM3H1L0+ldGeO6GKJfgVdL
xP+cFskkMrZU0sWzfZ2tUFtqx9MO+4YUmcohZuqF2Z7n9OHpLbM9L9IHO6P+Z9CHczNv5H6UEpvh
zCWgJpW5PDqS9OT4o0BS4dLi1jXz0vn7i53ZAjXe6zyFxGEH2+C4DdiE/F3XtiqZ9r1HXEfV/xu5
8b1nedL7rv7uFVRxvBu8z0zieu8hJ6TvBtv5E9kH/L/UH/b1/uy0OBNZb56Yylm47PV9dhpy1TKz
tnxKDKm7gjLYahp6/cqwde9mMQqS2iBI5jol54WpLkTcGK2Ui++MOMdkNptrSLxmJeMcF2yq4jFi
FmYyqIIikQFoSseLE5LPJksEff6xjMZlJ62v4PSUSG6SPrIno9fIJMoQw/c0IOHmMvhZOlJJLhI5
g1M4mF4HEEnnEySaeG6qojO+EV6R5BfNlQaZDTnPmGksOggWS6WjNnw3TzUmikDcgJK+78h4DCEj
q3qArXldVISWPGaVXjvi00gNpLi97yWbOFMrJsidT+HuI1nJxFrRm0NNUuD+SH6E2hB30/UA8lsd
p+mwQhGI0ztTXOWFKcjFuiqYrmWje9pKgwzOco9gz/tOl1Ck6lLmkb0xaU6iCDor2JpWOWWWPK6v
crdCY9QieX84Z08742aPxWGvcz9Ud7mHZ0cSh1FlL3A2OF5TOcUru007ZfqtZPtR75cGlI6EZ+fE
hUNYguP4FAF+xTNOcinAO0saE+G+yJvqs/OnyCRMA3GPIfsRsiJyM/HdNU4Lx8DfiBGqKtoyKXAn
UnzYOBhialpc0U5XQ3UsmpsWk0DVB5yvxnWCJDn7mGI1wUV9sUUQ4toiSjRdJAOAxNoorA59ASsV
7giti7AIo4ezvFbZ3ZFNC3epK7eKs8d8mB6vLrDJKB7kDArijnby8Bu2t7W8rGtHxXqlfooqsIqE
H9CX6rwJkmg27WOOI6jy8XH1zZY7KftkssMtxnJRR7F8mlZx48k8eWRm87cnOpdEWddvAxLG6acw
8znDaidKEFg31eMUQjYg06kzd4bZwe2UJ+CaibGzIaE4OgDaW60tpbf0H8kspO70z2aFHp49lrzM
eOUimDZdWBvZVadDj6OdVrKznrR0ztkbESNpXGFFgxhucq9lHX6BWT9md7hzxqUb1d7gvsJVr6vq
KYfJaIQ8SjY/tZ8jIvSuZHAFyLrRkjbB1bT7xcVNs7GeJQDZsKDXH3vX9mda98Mg3MbxPFU+ZpTw
7TY3N8pxSTQYxLNFtrEa/B9+sw7wu6zyPwH8szZw80PzB3OrQa6LJOxrOWuwCGY65q0w2Jh4uc6t
3A+oEnfyKc8V/N0isQr/O07VhHlbbLMFWdgBiIaTUVdhQj4hzo3ZOjmfvYc4mu0npzZzVaMt0k4l
NVx1XWqvyVw2qfKy8CUB4bj9ktCyF/Mogeqb1WI9IhpZuMl1lAjpPITXeuH20NDy+t6eUKd1t0re
xsMShRmmsTC/Smoqd0sLaCsR/T7PISlOpt3AZInkLIDRmeYydrNJOukXoRNlLmQeZ0wjkXBrJpkT
mSeSHJPjGB3xiHhprBtsqZecaqxUFOuUXMMERgwC/+WUPQBsrp0HpUw7NCfxgjhPsliMWVL6UJpV
E1f1ucKBipjpJrZckfRS/P4f7lTUXGB1XqHf8qH2vUqOToEP4/3lje3EBghzXWRIbbZOZ6QOeOJH
ObOc4u+05TFYoeV1MxLR7+vwpsyktTsmE0h+U1viGh7FxFikGYyeBjAhLv68B3QhMHPpJHvkhNnI
++mo+W1mQif1NeebkZw+fyI3BBeibjBC+uRCelW3aDW1O01mPZPtrVTImr8vi20WfypkKk9xP6hD
p7sF4sFaoYvHEetivjn6F9CGnBTYycovDoJjXhGkQ4q4LtIFuFyugMrV40AgtnmXOrxx+H+BjrGJ
hsNg84TA/7X/4V3eKgY9MElBSbhMkcTw66r8wT3abUkfUJRgNRcfURVbGFfQRqHbb3k0Qrj2k2/3
/BN2w2J8Isl/OGXGLVHlnz4/uOiUluQ3WEEvtQJ7oSrpIk1xaotOOx5HswxynNPATN6Xm4CBzJtn
tctkbZmeEn2PgQohczh2vJq3zfWsktqE4Og++XkIBbDyZ5PlOJTdU2aiuDs4eQBX7gxMDen0CcVv
dzoI4A/YB+2UXeENaAiKorcdNxmKOaYwH3Ik/AsPo26EnfbO41bwCKkB2pzaNjQhOttIJfCoR8P2
dOhi/WV/nx91imr+gmq/IOvr3vsrAVzSWDWEf2oyzhpqOKyj53J2dvMqx7Io/cVJmmax7zWlSf2H
FsvDAi0uVuxjyHJzn6sbK75GUF86z2+lpjocxtkpErLSnbgUck/kfJZOM0RbEwX4KZ2yEdh6EKv3
wlwKwSZInnh8DIUMG2CXk36sJmPm3pVAwNdFEz4nmbp5EfngqE+hHUAYTD9cZ5bPpCELE9bhuh4g
Se1eSV9U9taCcAQMuul6U3vlkmzTpKk9FrOAELVNgCS9VCDxH73keJrO416UDZKkx35XPFpRmvRU
Kcn60beOmiwcnBDMwPyRn1yzUdCOJTXasRU5IqtGz9MgXmE51gknOiP2mjNWhagXzOWddr3E4prX
tfEn51IpeiZVaEDZnk0E5571d8+PoEqTtf7WFze5U7HJ7gb7ElGfLrflXkOFZ+BFiKtwNC2Q2IQ9
9T9b/etVMy/x4KwDXpleoRQE5TCEtRrk7fZOIU1Nm0AifIKcxmakhxXq5Xmd5hiRBqxvzg/nBslo
dm6CNz89yNgLsD8+fXDwpnQFjIK4RaSfiFj8xEyB/nL5bRMfUqFHhnGeTq2QAS7swVvgmrp9Bq+B
agMzvjORGuAz5s3m+nlgfIV69SaM6Ab6yvIK1XirNggPgRWEKMtIEpPtNVl6ir38gN7dOfMgvOr2
ILnsEPgKIhr7pfEF4sXkF7ow83yMwtVCLmGhdUTNjuN1JFEJ4URpkUSkJisVczst1aC1QhYn0dQR
rc2YEtuu/urz70hSe8nNz6P5zBR7YLEbdUeMdzzR+yl7mtW66rPaeh4fJxkzsuJpbz3WvJnYKXB5
E07YnuWEH/J8uxkEP8czLYDCDqfJQvxSiUQ8C3a+JK5OMPM0TVDv0fUUlT1z0i9tbpZxQWV2JhOP
l5+Zz1KSMFygoZ6I7DFzaPtsL9ip0L1ybYw6DCbZVTijelmlysexVxKuQwy2FWw3m57w734JxdV6
6K5Ti+4wbyC7cu/9P6F3rK3cu97jkBOntsTDW9L8tKoxlZTxZbtVn0Alou2hUXd2N9bKqeLVPCFM
RBvL1z3Hzy1FAPmTrosMIoOY9UholuYJbFiY2JV5l5sH8OCmBl3LN5UYGWDgQXWuF1I8bQDjrgm7
TKZszW0hqS2ITT3wcnCA9dVcF8R292a+66RvgJZkI3ss3hcszPGCXaBs2SDj/JrOCnfzPJnnwS+a
5eNYM8pz/y2vGlIFVT6voco1c3D/na9F2W4zFedBAFrJg9VYWqOTZyFxMMEV/4HMqFVwYfOWIBt9
yE0rWplVSQMo25t832yxpYpvgOyT6bLeyOxhOZxIO2FdvBxwUSVQ7CzvyACEdlHXbVEONomIuAhF
rgckgogE8BU5iDQ7jKk8UIFZV81SYA3kDt6qm5iYAfYK8KIB1oGCysvcQB5/uBGgYA/na/t5D6kK
W0A12zcZp11qVMdDokRCMqx/hy7r35626pwO8RZeS1zn13ECA/XMgF+QsLhB8hs4BJTz6xrLtlq2
mOGS2EQxazQbOdZr0O/1szK1ftpZtGA13vS4N1R9Xo8Q7iAOfeVY09dIlDquDMP0gYNxps1bh9Nc
HRO6/r3/sP/yl4NXb38MzCkLk6uAvsbUKuC2DmkbaIWzo6mL+O9p/zcHdos+7qqJplYV4Tfn0Aoj
kuXenpcgsHwNqoyy+R3F/zMpzaM5CbX8Jddm54SFKHmdCSU1o7aqxixb6LwqcH5N4M3ztvu2JgLZ
L4zl1QXmFFj5WxVMn3nuC7fsgje9xK0U7Me/aNU6uj6TFLKKOwTzF6V5lLeCb9+enBzf3konIADU
RqUvdVVd5U3pqmbvxLPR2zMbiZoX8VqrqDL3ZZQyTx7JLjX8QNSm681IaDevfGaqn5176FBKoJ0X
0KC7lZ7b5rmLEd2yZIIWeB/czFKFmmS15gcu0JqF5+3FYtxTKb1jvNOdHsslyjRROVIcuPmwqsp6
SdOyv566ChgrXV7IW5XA7IyA8jJaQcst6KUaLHg8asGwV9OXYzh5mvRkUmOOFUB5DTFT06flFsWy
Pq3q/Gr/lnFrKruhtujJHNpdSK7jcT8icRdhZlxpTbS70RirvZRCcek0t7lK8VfxGRZEJ/WvjJcX
3b4WWP4Mwiu4WhI2pWqO2JUlRxLYUVNWh3PE8g5xM3AetqgG4TCmgNLSKLBRXPgBj3g/+OgWe4Py
ORqfw+6tO+9UGvwqgz0tjiZB+CmvQ/up2RWhmKOR1ZSdV5XiKGaWlGlWUtVHIqUCOz3qFwkIoXiN
B6dczlh3sSUl7PQRqhq1ZF4mtNApXNXkKFFYklyTvYnpw/x1WFuFSGzxwqd5+9UOgt+SxQmbAHDW
QbacQUlgYkpVcwBXX47w5vNfpLwDLmzIxNq60ftgshW+ncp8YmSS+nzWOYiOFLst+XbFxDGnXc/v
GHad+E75jCmWVghcnBMaQ9i2rnYaJ5yrwVoPqF+vvt8U3sqXucoDbYzfn842YRPTGFIyfvlk5eVP
8G3A2Zk1/mZKySqNZV9sgFBKPUrSCCkQFaswxwXFMnYq5HJTWhCvUAjP+iaZWmgab69r5Nq9UnaP
K+RVVUYTdMdLzJpuVTQBWQExKYpmLVHsQAmMKiXQdIkJh/PZdWgalkwqcaVwhcQgbMQQ3XYfSQXz
+njtjS8OCWwIKxM7NJtdEmZM094kml72TMaMrHkE1slJzalFU9U40zJuo+o5sJGzaLlYbCrDekYg
qf1XkJed8q9579XWoo08LsBBsPXVXGt8Tm4ttuoK7Ku07LlyrOTlUnBsaafzsF5F4XpBDePZ4kRY
FvZ/TGYxJ4Jg+SJHfLnosG1TTFTKVUadw6p0XsoFfjeZfXJ3JNa2i7an2Qr+15Xd1ptgwqqeRjG7
Yf53yQ2j2LRaK99yBV1Z+DOEK+bc1BU/vckRhhKGkFFjKyDEM76EwXS+EOjXkoHNRtMRrspoEX40
jJQUm2pWI0G4nDmj6TFaylI2K7JEDIwxcbAYe1wla5hFCyyUgNhZIQeSZUSRjggBQlcJk0vFTeH+
USsWfwaqKiEuceaWtJOkLfD6wlrcgRXXMkLDvky5cKMk0AuB+O02nKfLMVOpqW4jEA8oVHOjKt2a
STC7usyiGDQGziMph6hfryp/V2Lh/wnfHw4IS0hE5oqw1R4kbjvrLVWlm8uZtMB6hZh3ki/Y85ex
zkXsU+SYBtnDyQnaqrEQ/sH+rlxY/lvDYj4TkyE/rIwqyeI/eBJAjhVRKUOpW125DWUVI4oQ9gRG
7bmvOjkv/HGL/om+VrLJMOc2iocloOILDZGnWIja25XcA+kPFTm/VYRxVQw8zqtHctKGlsC28jWm
+nJ2HnHeZ+cyTVjaZKSgCNEUvAXL01K2jXgAQTTQK3CXJQvAZ22Z+ZhRiujOTAcg/0Wlk3rcGgzH
HjHFfF+qtBIRqjiBD/E4uoiHTdjSC8Lasz1P6VShz6mIrb0pDi1w79XZ0rykHfbDNQuFymdF8PJ6
vnnox5Nhq8MXcl8z+H6xvq/qZaUDn9dEoi2NAVhVjsFstYbMxGBs5mgB28OR1vnJFgqKKEAzzMLT
h4PsVGEcWXEP6bkg58SRGgpZNOh6xXQN1TF5m2KncndF/Z/7dQrPisOyPrm1lVfaJugzqK3wMmvr
+0Lk72Z/rWyZJpmXN1SnteI8aj0OK5wVsRm5/OMUDb7VFc/ND28uXfnas6J7j/H3lw6XV1Df09ty
yRl8rlxRvY9k0U+yn7BHXp5nunddcXQ+V1Zrd1POmJmnkBVLHmc8y71DGBfmYRusyC2fbV1CAzOe
iJth7tvJHQIIm7fZ9gpbIqA+ry5RAWrJMP0HszM9CNphru0JZKW/Q0/M62oZ3145jRZrXmjIu9e4
sDTRoJjaIkTVSQ5uT2fgMrDsd5wPeXW721pVkQ1lgcrwSUPpOyIsj2vDl1ZhTqCrEWj1H+3hHKGT
7XazGlxHbWz6rfWmjI8MoImLJeMswxFrRZu7AeddEveMJYKe5XpGRq2DfHmS2rsfByjHTZf7Tt4/
vsmywjM73GnewTBnT3LEJ9mepTOolqcLyyAUSYoJ/uHErbhMXVEhOLW8VUOiBpRM+KYqa2JeODtn
GFs1+t7HnVJm0dsY0xUboIWWXI5T1bOmVD1Mbka6tRo7jAWlkuhIjVbHF3fmcZ7iVWCqsIm83+LY
xh5u/iU3xKgO1VbCnevPtVs7m3nJ2XyFh10ul1RYxP8UY94KjLASK6yFGdYvdWOwxPHtWML4lq6H
KeqxxbFii88uV/cZWOCOXgkSrnt+S+yEhkLQm3bBqDxqCyljdY29qF50gXN7qQOnMpwgByNROxzT
HNV39fFq5sgTyREv4XxZ8JFQGT8P5gBIfXj+cb+Hgt+9/V/3P/xnGQ1i3iziSD7euTrDD6PspJ8i
YC47Sc+R2JaYSfZS+dGE/oWGKefPJVNbIrp/DhFo3kZEpTqkW3jRW62NMKhYWiHaoAAit+k7KjfX
QxAel4A9l4DRexUOKAQ92ASSaUYQU3dXxlYXhbQ7RlljJvUBpF609XqBpJ6NN5r35KbUyj0lrqY+
ClUl6FO4W2rPBS/j/rUEon5b77nnHE7xMkF7EXZulyz4Ii77k2QRMptqt5BEZtmj/CJ6paR2qzKM
3RJ4Kytek14YS7+WlxqaWPPuF53tpxc3fjx4Mdh2NxgmGa5rjAgEu6jV0jddd6yiK4VAkmFwkizy
YVhehBKYER8b86CunQahilb9mJYQa9uFFAtj2XuA1JEQDZpdWHNQLnWezkgU2Chv36ituqHVyiGr
myHwcXEe0vCXkVi2iFV/7NgjoclKz2mLFCexLj8w6L1lnSp2LgI10Z1Hl7u50W0AZSKHHbF1rsj6
QcN8wtY/jlI27KQELrkW21ms9nNCjhJEMyzLpH9mdJofoTZqr45RQ+2KzwpRc7yMYJj7gNDbFcG6
YE0+Yeaf1NDplNrj3KNG45j8wxiXxYZJ4gfdzVxd42SGVb/rlrqb52k92dGMy+5KxBc85k2Yy5j9
jkDWhKpJySKTuN4xWNBNp6fpaCS3L51KglEvCkwqGzshxP9E4PCyPnD47pHjLVaC2UcwK0o0uT4Y
IW3JVX38zkjSmvhKeWYA9mrCkqAgoduKYV2WMuDw2hycK75U0GRgBizmZiv305s7xge58n0KOsha
wasiCbtXGVlSHfGWVhGqW5nRu0QsrTahflbIEg6tsj7kozvEKeWVLPCXjRjrfvsNopWe/a39UOOV
Ao1YwhAPDt7YmJSWY31V/DxcJ6KpNrzCar7WjWrq3yGgaf7ZsUyrNmqdiKbqgCbMTd05c2aGsAp7
aKrPl6AL4kScNJE5d6PvS576udeB2RfPn57Q+aInZN7q+JV9KnJPUoAz31uCpl/j+TBBrONzTmdD
nDt7y1XAGPxa8U6YtE1utip41PjNHrx7/ev+D0JJTK4e3g5holqaSCceOgVcTWIuLR0hfHGjpW7T
zcr8Ye5ivkeB1OmfvpSK6YNlEE4NnrHGgScPXzb6G2HH3CWm0zss6AB2vIqVOKazbEGoUG/Srcr8
yuVJRy318TXEHfwmnJSGd5vxh/h39kAkSTrK0qkcQ/W4H/b/ff/FR4KRiv3tIq4CHdw4Y+c30IlQ
rOxaXRJwR4SNScdL1l/oDfd7tJzSa1SuyBw3M/GPT3N2ROuxTzNqCNCN6FKOMTzvE6vuuOqy8ZmT
9tQDCZVLzUrjgqHBFCfxhaYddS98oVyG6+drPqigq6b57Q69NfnGBsu5SsQsakKJHK+SNal9AaMP
SIIkqWtQhdHvS/dqllvDU555WA1FxJ5aiYPBxnjn7Tr7Dd5NXAjPk+kQWhLjPjhpqzetYHwV8dgX
QD6wnZsTNFpZPjpbcPj2QKsy+nZTXidsEOtUOYTQfNf1CUHTlW4hJmtkOi9nXdkqqqAqfS18PeCH
/ffvPnyU5sXMe3dSBQRfm23wNGregr7eo063yu04af1wtQuCa9kHXNIkrn+/Dn7HrXf8fuxJVJFW
HHXIZLVMQv3cpJ31E5Jy+hBzKHfRCN4SpsZgnLMUqyrNrwCKdTTZ0zUDznJAvssyVeaOOeDMhwdX
7VjRty9/l5BUUqkkrKGKyZApEM3D6kBXK3DWuLd1YF6h3rRo779QmKEveEm88IoFYDjHUTpSnk+K
BMOt/XIRi4uRao1jOPPaIiRcg5g9+uIprRwZKTlzSlmDlrOtR4WUZwbHOOUkgDM3++q3FJmMG/P4
rCj+9asiBjlPvZO5vtLUrjE71Dbot735ZmFN4d+p8uc450lWkcRFayYeXKKkFVKd6bkVLQ6/vH31
Hz3CgC9+Ui+B4XVg0+v4BdxNkQynzmC3s3PTdf4fXNFkuB46hn8QPHzS6QRfBjuP7JMn+PtJR//G
r1KW/ovDweg4RCBf82gDVXrwqzG8wSacLUkq7Xbvqw2SCRfRSIkJkdz6CZK8F5O5S2mXchWPiOPp
TTGZPKOXNdKcp8575jN2lk+bu+X6AsiNyZlhi0DB2nUeBv+/jy7LinBgak6MX1UVhE+yOI0hnQlu
SzzsRf2yQh5jfb3n1v6gLqyOYysrly/QrxS8+McXh9j/I7PVkPB7zn5nsKz2pGxOZeXo1Zm7zfb1
THmGTqeGbbso1zeyt2Z6jFp045Dduy4qbAwXhZ0rFRS6cEsHVSWRlXIOOCGpT1QcBfVc54RhkWT4
2z2rcats1Yv/uBe6MLup3bdum2ShO7e8AgautmYYuCqTPzNpDXG+vg7oV7qkQks2N3H/AVdc7ohm
1+AaQcHV8KZRrf5g8L1XXx3EDHguFTXa7RA/H0k8OHbgyK/KAxQkVXk+I3dALfjGw2MDsn4FN+zn
BeeItpDWoptiEYUHgkGEOj6wrtOLm3K7i9HIaYoKWrSOi86opnmp28fcsKp64brQ+KfA2crdpPY9
6v4MUWtESFiVj2KNGdci4N6y3jg5tSbDKvRbLnLCSzh8euR7LK6oiiIVUL4pfODsQjErbdHRMSJp
OXCtkwZSxU5Yfl7uAA1L7pPWpb5q+3wC3+NwL7G7AqfG06G3X85qqmyBDHAtQFjniNbytLl7hy87
Hf5yG1927vTlUx7tG7vOm41/+7/p3yJNx9kDt8RlOmjPLv/UMTr078mjR/yT/hV+Ptx+uPPYPJPn
2zs73+z8W9D579iAJZj+IPi3/5/+++Leg2U2f9BPpg/i6Vkwu1ycpNOHG41Gw4cJkuTfvf/46t3b
56+DGfRsyFDJIsiDYBRN2VgzT8dW7/JevmZDvGh1wre/vnkNhUS6PCZZfXE5Q45eRA8ms9YG/fwP
YpTnbGWhP6ZnyTCJtrJ4seCirO0geMvlKudpijDWt2oWPkGsxTQzWb2gTuIvpGo28kUuoWQAeios
iA2VweHW1jA+S1A/otPaPio3Q5XHSzST3oKXr17vH7mfvea/ZE9+u3775fUkurjWyr9H8uE8Dt69
fHmw/7H35qf/4meTeOI+qnMPo5bYY+nizau3LVQesk/RifsQ5/D+xcfraLlIK1aCErILf+YbG594
fZ9E2GQ5A3Z98UpiqzAn9z6JhkEoSxwnE1QASkcj6i1rsRiKUF61fWEOs3ScDC6bJqiYLhhLs+OY
7ckbn3gin4LZkqSePE15tJBw4zCSMW2kvFHG058PnzwMfguOYTU2ocn8qMUuHItU06GihulloGfQ
bG9IwXGJUANcwBCPiGJHM1zImeYWn97lYAjof4cudFn3EtjIkbB3BkUiO4fCmpBtiBsv768LsDEd
pTg4ZEaYx3MxPnBQPNtTTRNAKdd1xzD44yTN3SA4L3cbt3UDHgLzhd6rVvB7BleMFBldL5H9gsTi
jY2Dj88/7sNgm7W13jDyOoWN9x/e/cerN8977170uAkS7D2grot0gU+y0dx48e7ty1c/ru5I2nBP
8WJQ7AmXiTr6giALrnDRNFlcCmhpVPXMJJeXg/mKHp9P4WBD60sl+w2OvB8HnEyUjhL7JRsawVQT
0yw/7PfkknGFwsdgAh52Os2NN/tv3Bf0jN7sdPDq5fO3yEuFggydjY0NgiGpqTw9m4zDZle0AhFH
0a5YPFBdQ/z0WJ+zl6cBAWRxjXGI04fo6khCmqhPFlwPj6BePWyMk74iwcm4naVtLvpdeNg46uaq
3vll1+ekEzCSAhDtFz+8fh2yGrTYpo2l7UOfKMFxbaTup2/ybwc9AsZ5b+bLpDC169ev6Ph6ZzuF
ZEDw8GNZqVv2D44SWum7Ax42bJg+rCH3y6xBYlrF5MJ5s60mlGazSrNE39jH8QVMlGYYaB3ibsk/
HJWeJMr8MkPR64XVyuLcAyZaobpfvv311Q+vntssCeK+Hg+/a9oZU4dNAhvCMESMXsDrU/JkEBj1
elDh9XqhJIugibaCZHjRzKeEF21+zr/RS9ljbue3OnFP5yxNhr2Zs/vu2fzAyP7HePETIZBx/P3l
q+kwvsBp2e+XyRSa8gvic/VZ/xLqdxnJ2eeaE5XT/AD//kmsRwq6/6U9SfSuq0ICmjnqbduNwRPZ
k7mzGXqeZk9WAULeFaKrta8Rob/70ZxYh7xPwpNhetoKJsQqRMdxkwskT+WM8SkbBJOM0wxml5M+
8TMhYpvlwJtsGZ6KX7npQygPkax5lJ20gYitrA2v7pgw5kKWJyeLafmhSfDdQ8942q0C6ZfIR0N3
HwMlmi1bQTD8Mmtif0dT7+zN0ZkdKG5qOBedQysIGw05VdaPMP5xTynfWWC4fGMZTPY8AKKxZvPY
2ewzA6IOYPLmZ8ccjUaj8HHxQc2QM8sDvrNmadpnbSXhI+pIJsto1U4S+I1n6UzDwYGDORwMNO6+
11+iblH4tyf+9Hre5BreFXpL/RMW7rcC/+5QH81ygRCerAVSd9a4HY182sxY6bxJhg4mv3VtRKWy
W8rLtMAt0P+iCycBdSs4SVx0wFNqFue4/irfYzpvoilBN1J2vMYMXrCKJaKO4E7uHdQ4LeKNk6S8
HQKRTCdvHQzJfddr/oNsi/mqwjZUDTLYsuo3DsgTK5sVocnFSuXZvV1OXtJHDUIs86CT95RNoxmJ
G4tib7NxK5ghuJdPdUbHao5FQKK0i1cNftGbnDe6/HUDkkGPmHH6u2ZSP84GL8anv47eMcfecOCC
TrNizxokVqzu8U08WdVj0MDOmc95F5s3ykvNonkW91QACbNZPCAEqjuSjBiX4iH2j38SWmoQhDZK
JzBOskXISYLCqUKbVk8/ygviAV2im3Y2G4Owtxru5sODn+OxXdKpk5i1k4x1wSGfJS7QjM2fUx9F
50xDn5gFWVfw1ZfZV5oCRHlRdsf5cghpOMyajLIRYjp17gnNvs2y7DCU0eSVLpfe6g6yEi0kWqJL
GZSv/kYtD/CCNXA57Zf7OtCxKsh7vr5yP3di03QdA7l1uhY2lohcHYIBNksagUJciWLWY2u5kjj2
M5GKsO5XbuZ3XjtyWhZO15jw0cRkXtOGSObFpV0aX9D9xcvGXgPjcNK50i0hHHpmRxHo2qObsF2O
EqbVHJ6amUCBe+ZNy2dSHfAkDtLbudGxvUMJUYsWy666dm+TkLIZFwGwcNHMdf6Zcw2OXICPuIIJ
4BxeP1EVc/crTo1nWLyIGM1dSN6yCoq+zLjQYGb0FpjXm5/+0Qri9nEbWao6rZ1vOp0AZRX5Hsld
wVKJJoAh4S2YxvEQyMZsQ0vJYNV24JjQdET8/AK37zNmG9Eo6diUMgHbSJOun6SDSMYpNvcM/ztJ
asb4kkMY6IrDo1d8pJBFC/5NHAfx5bDd/nLojHFm1+tdrjODIibDHrRbIXOchJQMD2zNaXiWTwYo
BLKKtE+aq+nToEScjgmJTxgPVRCLF1AMoShzo8CRFBmUDiwU/1QPOw46XdRM52M8mcVzpLqN15hQ
fl7cnaXjTY/LH0eT/pAkh/Nu0PiO+enJueHoleGjI/4NByixIPTahp9seJ7gRnAKrOCEPWG+ttks
tcVvuVKOPoGXuLKK9FfLwhG9abfp/03tdRTOxsg0H86G8oMOWH+JLko8Acb5dmtLQxGdEBwdipVW
HLwBHRhvwEwSFM5EO0Z/fYvf7wed9t++0S1pVK5H9IgmAGAeY+qMIDRhnvmTfr4A1RQ1ZNe0ddrx
QgU0WwEu5mHOLB3xn4bROfJFXHrDDMyRjwhHgIHDKpginu8AwUhEWG+DqdOmIORT3EJhX+xwTV8N
nO8JXpunXw45wApY40teof0cmraGFMhARRwX84+ypkWdQA6sk8yxA9cKUBQhytpq6mveQbcocWVM
g1ll6KPVUMlZy0GwzQraBqaJSM9nISekNuY0OmHiHR+eq6jME2aRymaLevnqw8FHy5B3CXVu8W5A
5LdaRcR6sNusqHPnxAhOSQj3OT/0fUiDHdWhhjWx5yICt2PuvZHovYufJ8GVaGjicZ28uCNVRStx
UBm7wBVN0yndrGiMLS+yQ+lpmb/hAHblR5nR8BNyINqxnLop4qp029VBLgDlewS4YBuNPt3qh2gT
DCXH3Jv5+mYInEUtC753vHlQhDa8M5+dd0s5omjXq9UqQjF5xxoO+qRejabFMQgcJ2ccIGqU02IY
0Ei1xjq7wtNjQYYggKSiCRY0c0X4/Pu8raJW235Y17pNJ6SM65ck3khzbDfhdrB0jNEJ8wr7Mzs/
7G5tHzWF/rjkx1tE3ov5DF1ss4a8+AE3RYSZwDkcXTAm/XbeLLfONUCDku7hoEYdUESilR1XnGhQ
T3Xza+KCkQUzEIpG4QaBj7R8J5h6aQW8KzY+mud9x+bgT/GWda8U0j1dmLtWJnpCAoMvvx6CMmK5
Z7esjuje7YtDI1kbfqOl5UaTO61spbKgdmVKxe++NrAPvarjU6mIhSdeYN5SlmkNrY07ntzyNZtA
mU3NSsAalTRx/VWnqfyPWlUXKcHw1pd29VKOZ50tqDjj6h3Iz9nYlBt3PV86qT9pC/TYP3MTYHt2
aAX9WaYU9LBAGHlhoqUq0pApq8XrqYeMkOvneQYO8RjkjhElUiFzEQXXcpE2uhWoUjnOil0nlvOF
dPyeje2NMqNZIc4UWc9p82jV2roBZjaJFskA+uizmAsjjMOU+0lb4tRK04TObzdQ5nNi3k70rRGF
0+Y65HI2WLi4iKbBwKlbLdbZFtB4s27Djiqz93jL7lY6XPCca8D8MzZ8u4JKGY6rW5sQ6JZJ1Aka
FePTTtbMgDbK8HYhxltBTgGjUg8PUsdQhA7q+U+DBMsbCmeYc4O4u9KPJMuDl3cuUJu0EOImMTTC
5GICta74OXxN39PfjbIgw1GtLLvQexrmvCGx8if+sbCcM1xOZiEz+yQfnzj62qxNezSOBrF0IlLQ
xq12Z4cXVjtzLmnknjLgkSEKdIOv2FvmK5rzeMzRAYpblEGUDePBW0Hs64O2OVaDmHLeso4jAHKf
JfXQv0bYK0iUdxXz4AG0J50wWheBr+kkfy5elA9YXJEs3/pBBR1rbtQgEEMvCnjkT8TUXiwWwiWq
BJkvbHIUNBHfLeQvl/yCEGsRHSdZZ/ppuiBoGqZ8AY+X7Cy3sK581p3r/CRB+gNOQ9lnqCwMSSCo
3jjGGSrbNQQbRTaWUyXbpkAFMAgM55aURMdRMq1WdThCMLsGZhEimPX2uZx9xGFTPGeJ2I0yKZOA
KYQ6HTMV8UqzE2iuUKy5eSc2Vp7rao6dDiSXJFjJJDbBlYB7K7dsezWaq1Kn7BW3F9iG1lzoQRRa
rQG6awljnJ6oAp17RwmXIuTn8ZwIgeN+s56E9NeDLzlX1O1n5/Np+UG2VNLjScnWuCJf3aE4byo3
tgZlCU6apTPFSa5jh4dKoVNkTdRq6pLjVyFJ+aYyt0gEB9YumA2yRaaI2O+SaRKi/sNaalRvYDL0
YYL0htH8+Mw3y+IJNgU/EcAB97XG1kmDubOTeDzDb/yzWSR2vd4wHfR6zd18KIa5yZAzbIHf015b
8st298gYAltqVBUlJZPGPa7a12i0HKe6cXQMMnHVUE/cRjfIdUUqnnfFWt6wQm2Xbd2NgvRHT3PR
0BeMzCv7FbjSLtu7G1pY3KozJaN8EnyLgBQkJlw4+4KQEDw6TAqGOChreS2c1wohRF4HbEzkBofR
0ZHtBO2OdtF+L9jxoSZiCcP6/DbqOjY+wXfoUgy3jRVT1eNa3SPUTbk5bDnVMoYwug/SyWwcE/eQ
ziT9g/CHUdOpQ8xjaB0zNrjQlyGBlkhWUO02rKk3V3Mz0NCXPbE1u4Zo0yf1Iz6sjvrVbpL5DcmD
tRvGBA2zz77gWdfG9wSzXIaj5zRQqCDnQaYLiwDBZkkJyr3ZdWIbzB+Hp0f896nRldOzbrXtGnDm
fGb2PmWUGS8TRPFwSBz6w87THw3YHxtV9k5m2MGzeFEEmYQRZCz4iZN/l+shMKueLYepcadlp1bH
JbeMboGlWc/huprkWMT6T/jaE4UYmXeVCbWAvQrfCaSVDkChrlunErdijA1VYL7n3QvLXKGsmOXz
oHc2Ak/FfFx3gTozT9XkWRxodN1vixJCxUmaq0q3dIKjF7cXDqu41rCJawk6YKpNvSo5hTFl7ByY
cSA5OFkuhtQjPDo3EnjMgqz3ejzHXg+EqdfTPbbTYHKFvwzloHP9t7/+/fXvr39//fvr31///vr3
17+//v31769/Vf/+X89ItToAgAIA
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
  oc)              cmd_oc;;
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
