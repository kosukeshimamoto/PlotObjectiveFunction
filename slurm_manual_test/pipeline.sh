#!/bin/bash
set -euo pipefail

function prompt_with_default() {
    local prompt_text="$1"
    local default_value="$2"
    local user_input
    read -r -p "$prompt_text [$default_value]: " user_input
    if [[ -z "$user_input" ]]; then
        printf '%s\n' "$default_value"
    else
        printf '%s\n' "$user_input"
    fi
}

function prompt_yes_no() {
    local prompt_text="$1"
    local default_answer="$2"
    local user_input
    while true; do
        read -r -p "$prompt_text [$default_answer]: " user_input
        user_input="${user_input:-$default_answer}"
        user_input="$(printf '%s' "$user_input" | tr '[:upper:]' '[:lower:]')"
        if [[ "$user_input" == "y" || "$user_input" == "yes" ]]; then
            printf 'yes\n'
            return
        fi
        if [[ "$user_input" == "n" || "$user_input" == "no" ]]; then
            printf 'no\n'
            return
        fi
        echo "y/yes か n/no を入力してください。"
    done
}

function upload_project() {
    local source_project_dir="$1"
    local destination_target="$2"
    local destination_project_dir="$3"
    local delete_mode="$4"
    local -a rsync_options
    rsync_options=(-av)
    if [[ "$delete_mode" == "yes" ]]; then
        rsync_options+=(--delete)
    fi
    echo "Uploading: $source_project_dir -> $destination_target:$destination_project_dir"
    rsync "${rsync_options[@]}" "$source_project_dir/" "$destination_target:$destination_project_dir/"
}

function download_outputs() {
    local destination_target="$1"
    local destination_project_dir="$2"
    local remote_output_relative_dir="$3"
    local local_output_dir="$4"
    mkdir -p "$local_output_dir"
    echo "Downloading: $destination_target:$destination_project_dir/$remote_output_relative_dir -> $local_output_dir"
    rsync -av "$destination_target:$destination_project_dir/$remote_output_relative_dir/" "$local_output_dir/"
}

default_local_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_remote_project_dir="/path/to/PlotObjectiveFunction"
default_remote_output_relative_dir="slurm_manual_test/outputs"
default_local_download_dir="$default_local_project_dir/slurm_manual_test/downloaded_outputs"

echo "=== PlotObjectiveFunction RSYNC Pipeline ==="
echo "1) upload + download"
echo "2) upload only"
echo "3) download only"

mode="$(prompt_with_default "Mode number" "1")"
ssh_target="$(prompt_with_default "SSH target (user@cluster)" "user@cluster")"
local_project_dir="$(prompt_with_default "Local project directory" "$default_local_project_dir")"
remote_project_dir="$(prompt_with_default "Remote project directory" "$default_remote_project_dir")"
remote_output_relative_dir="$(prompt_with_default "Remote output directory (relative to remote project)" "$default_remote_output_relative_dir")"
local_download_dir="$(prompt_with_default "Local download directory" "$default_local_download_dir")"
delete_remote_mode="$(prompt_yes_no "Use --delete on upload?" "no")"

case "$mode" in
    1)
        upload_project "$local_project_dir" "$ssh_target" "$remote_project_dir" "$delete_remote_mode"
        download_outputs "$ssh_target" "$remote_project_dir" "$remote_output_relative_dir" "$local_download_dir"
        ;;
    2)
        upload_project "$local_project_dir" "$ssh_target" "$remote_project_dir" "$delete_remote_mode"
        ;;
    3)
        download_outputs "$ssh_target" "$remote_project_dir" "$remote_output_relative_dir" "$local_download_dir"
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

echo "Done."
