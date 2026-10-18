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
  if [[ -z "$line" ]]; then
    kv "members" "${C_Y}none reported yet${C_0} ${C_D}(a member shows up once it has sent its first rate report; the miner prints the list every round)${C_0}"
    note "                    add a worker:  $0 pool"
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
yyf0dN/7O9/cW6mHzjve/BYP977dwMg7inhc5dhZTgMSbyvjB3r8DVCV5/92ETtwsPvYnbOCttuq
W7P89y7XZp0bmLZdjJ8sT/zDzbc3IW379W/c86/x+Cme7ny/DV+3FQHaf1lm2FLKGqV23jVXjyl9
K7VbuHTjPTx6vpw+3femR6Db3ny6nD7f+mYHbbbf9djz39yzHQPVxKKbA7V+uvl6LarayZtosl0s
6O6Wcjzg2CzWW+KUfRcMmDdlYw8i5pTESb/H+s2clO7canSsSSlejpv6CQocefqdCBr/ShU9c7YG
tcHNJXmz7rcAObLUys8D1pF75c4FVN0KrXi/AWFmZ22ATUuS5FyF5QFUsqJQiw7uWVw0yNIWtnQA
E61JdoemT6T5xFXnt43QiS2/m52OFkNfxRZsQ97eEsiiLYCCawBYgtq+Xy2c3tBto51l43LxfaPU
49XkzQRv+LDHCgnuQjPtnl+OXLxZGbPQKZzO3wNuOg56pQf+RMmZu7vbS9ezKoq6A3ich/2P/eM2
bwqWB78GYV3m8hJv/+nFn939XajAtYGWA5oX9lZ++yt5YShsyQWC+f2wP65mtD3qx416hEA9gmf0
r+B1bd1mUtun04DC3Xx8GxLa1fkKgtHy3K8/cFEuaZQlxCJ+MAi+/+GvQczevi2fxSl7sNGkshG0
+W5qXachtKbz227GlIlGe8vWzOWkitxcXspJeUPgYjaWt8ENEU1PL+hy5ekNORcXxsX3UfjpyclS
zsG3J1eLbkVDV83xVjFBiHkNkGnmvj/zBHAMFrybE2bGnHn/jthnR1yRZ5GFlwbtgCD1WelffbTL
tqJpRcfvDT17snXj3Fz0cmYtxlxiF+z0e252kgXiZFgb7Bx30TblpfHZbh25je/ElOz0dLJlxZ/e
1mfNZRnWNk7u0+6yxXpAp913Gn3cDd6pyUTna3wevGIULevl3Tmtu7z+8PodXuv3nDktF0OU7sQU
hDTKo8OGuo0iCLe1YWWj6po6gq+qwZVODoH9YnCzfIKGKP9XWfLIrw3+cm73Tbbq0u16K3t9ELlq
gFoO3LU78HmTTDVMPd4RgXXdkdbbRxoigxrrzJIPHNwaBpctxbeNhlZYeKORFP9Z9QAShiekMFqG
jt8gOgjSOHgYpPS8POEvcApjNKat8VfyJ+R8SIu6ZWRpDYZWS2+W18su6+WWEbF16bMmthn4tRVq
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
