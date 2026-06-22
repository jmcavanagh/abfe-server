set -o pipefail

LIGAND_SDF=./v2_1b_1place.sdf
PROTEIN_PDB=../data/9qdz.pdb
ABFE_CONFIG=./config_abfe.yml
FORCE_RUN=${EAYBFE_FORCE_RUN:-0}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run ABFE: ligand pargen, setup, MD on three legs, and analysis.

Options:
  -l, --ligand PATH    Ligand SDF (default: ./v2_1b_1place.sdf)
  -p, --protein PATH   Protein PDB (default: ./9qdz_fixed_dry.pdb)
  -c, --config PATH    ABFE config YAML (default: ./config_abfe.yml)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--ligand)
            LIGAND_SDF=$2
            shift 2
            ;;
        -p|--protein)
            PROTEIN_PDB=$2
            shift 2
            ;;
        -c|--config)
            ABFE_CONFIG=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

JOB_TAG=${SLURM_JOB_ID:-local-$$}
JOB_DIR=$(pwd)
echo "JOB_DIR: $JOB_DIR"

start_mps() {
    echo "Starting CUDA MPS..."
    _CUDA_MPS_PIPE_DIRECTORY=$(mktemp -d /tmp/nvidia-mps-pipe-${USER}-${JOB_TAG}-XXXX)
    _CUDA_MPS_LOG_DIRECTORY=$(mktemp -d /tmp/nvidia-mps-log-${USER}-${JOB_TAG}-XXXX)
    export CUDA_MPS_PIPE_DIRECTORY=$_CUDA_MPS_PIPE_DIRECTORY
    export CUDA_MPS_LOG_DIRECTORY=$_CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d
    sleep 10
    echo "CUDA_MPS_PIPE_DIRECTORY: $CUDA_MPS_PIPE_DIRECTORY"
    echo "CUDA_MPS_LOG_DIRECTORY: $CUDA_MPS_LOG_DIRECTORY"
}

stop_mps() {
    echo quit | nvidia-cuda-mps-control || true
    echo "Remove CUDA_MPS_PIPE_DIRECTORY: $CUDA_MPS_PIPE_DIRECTORY"
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
}

restart_mps() {
    stop_mps
    start_mps
}

run_abfe_leg () {
    local leg_dir=$(realpath $1)
    cd $leg_dir || { echo "Failed to cd to $leg_dir" && return 1; }
    if [ ${FORCE_RUN} -eq 1 ]; then
        rm *.tag
        echo "Deleting all tags under $leg_dir to enable force run"
    fi
    if [ ! -f done.tag ] && [ ! -f error.tag ] && [ ! -f running.tag ]; then
        echo "Running ABFE leg: $leg_dir"
        bash run.sh > run.log 2>&1 || { restart_mps && return 1; }
    fi
    cd $JOB_DIR
}

run_abfe() {
    local path=$(realpath $1)

    run_abfe_leg $path/complex
    run_abfe_leg $path/solvent
    run_abfe_leg $path/restraint

    cd $path
    if [ -f "complex/done.tag" ] && [ -f "restraint/done.tag" ] && [ -f "solvent/done.tag" ]; then
        echo "Analyzing ABFE results: $path"
        easybfe abfe analyze . --force > analyze.log 2>&1 || { restart_mps && return 1; }
    fi
    cd $JOB_DIR
}


# Setup (ligand/ and abfe/ are created by pargen/setup; mkdir so realpath works on first run)
mkdir -p ligand abfe
LIGAND_DIR=$(realpath ligand)
ABFE_DIR=$(realpath abfe)

easybfe ligand pargen "${LIGAND_SDF}" -f gaff2 -c bcc -o "${LIGAND_DIR}"
easybfe abfe setup "${ABFE_CONFIG}" -p "${PROTEIN_PDB}" -l "${LIGAND_DIR}" -o "${ABFE_DIR}"

trap 'stop_mps' EXIT
start_mps
run_abfe "${ABFE_DIR}"
