set -o pipefail

LIGAND_SDF=${LIGAND_SDF:-./v2_1b_1place.sdf}
PROTEIN_PDB=${PROTEIN_PDB:-./9qdz_fixed_dry.pdb}
ABFE_CONFIG=${ABFE_CONFIG:-./config_abfe.yaml}

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
        easybfe abfe analyze . > analyze.log 2>&1 || { restart_mps && return 1; }
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