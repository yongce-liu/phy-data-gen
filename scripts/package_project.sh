#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository root so the script works from any directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
repo_root="$(realpath -m -- "$repo_root")"
project_name="$(basename -- "$repo_root")"
timestamp="$(date +%Y%m%d_%H%M%S)"

output_path="${1:-$repo_root/dist/${project_name}_${timestamp}.tar.gz}"
if [[ "$output_path" != /* ]]; then
    output_path="$(pwd)/$output_path"
fi
output_path="$(realpath -m -- "$output_path")"

mkdir -p -- "$(dirname -- "$output_path")"

# Include tracked files and non-ignored untracked files from the working tree.
# The archive itself is kept out because dist/ is ignored by .gitignore.
pathspecs=(-- .)
if [[ "$output_path" == "$repo_root/"* ]]; then
    output_relative="${output_path#"$repo_root/"}"
    pathspecs+=(":(exclude,top)$output_relative")
fi

git -C "$repo_root" ls-files --cached --others --exclude-standard -z "${pathspecs[@]}" \
    | tar \
        --directory="$repo_root" \
        --null \
        --verbatim-files-from \
        --no-recursion \
        --files-from=- \
        --transform="s,^,${project_name}/," \
        --create \
        --gzip \
        --file="$output_path"

archive_size="$(du -h -- "$output_path" | cut -f1)"
archive_sha256="$(sha256sum -- "$output_path" | cut -d' ' -f1)"

printf 'Package created: %s\n' "$output_path"
printf 'Size: %s\n' "$archive_size"
printf 'SHA256: %s\n' "$archive_sha256"
