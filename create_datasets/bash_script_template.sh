
set -o errexit   # Make bash script exit if a command fails
set -o nounset   # Exit when bashrc tries to use undeclared variables
set -o pipefail  # Catch mysqldump fails in e.g. 'mysqldump | gzip'
# set -o xtrace    # Trace what gets executed. Useful for debugging

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

arg1="${1:-}"
