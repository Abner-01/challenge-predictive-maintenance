#!/bin/bash

set -e
(set -o pipefail) 2>/dev/null && set -o pipefail


error() {
    echo "Error: $1" >&2
    exit 1
}

git config --global --add safe.directory /src
current_branch=$(git rev-parse --abbrev-ref HEAD) || error "Failed to get current branch."
origin_commit=$(git merge-base development "$current_branch") || error "Failed to find merge base."

readarray -t python_files < <(git diff --name-only "$origin_commit" -- '*.py')
readarray -t no_committed_files < <(git ls-files --others --exclude-standard -- '*.py')


all_files=("${python_files[@]}" "${no_committed_files[@]}")

valid_files=()
pydocstyle_files=()

echo "_____________________________________________"
echo "  Starting Python code formatting & linting"
echo "_____________________________________________"
echo ""
echo "Detected modified files:"

for file in "${all_files[@]}"; do
    # Check if the file exists
    if [[ -f "$file" ]]; then
        valid_files+=("$file")
        echo "  - $PWD/$file"

        # Exclude  'migrations/' from pydocstyle
        if [[ "$file" != *"migrations/"* ]]; then
            pydocstyle_files+=("$file")
        fi
    else
        echo "  * $PWD/$file does not exist or is not a regular file."
    fi
done

file_count=${#valid_files[@]}

echo ""
echo "  Detected $file_count Python file(s) to process."
echo ""

run_tool() {
    local tool_name=$1
    shift
    echo "------------------------------------------"
    echo "Running $tool_name..."
    uv run --extra "format" "$@"
    echo "$tool_name completed successfully."
}

if (( file_count > 0 )); then

    run_tool "autoflake" autoflake --ignore-init-module-imports \
        --remove-unused-variables --remove-all-unused-imports \
        --in-place --recursive "${valid_files[@]}"

    run_tool "isort" isort --profile black "${valid_files[@]}"

    run_tool "black" black "${valid_files[@]}"

    run_tool "mypy" mypy "${valid_files[@]}"
    
    echo "${valid_files[@]}"
    run_tool "docformatter" docformatter --in-place --wrap-summaries 0 --wrap-descriptions 0 "${valid_files[@]}"
    
    if (( ${#pydocstyle_files[@]} > 0 )); then
        run_tool "pydocstyle" pydocstyle "${pydocstyle_files[@]}"
    else
        echo "⚠️ No files to check with pydocstyle."
    fi

    echo ""
    echo "___________________________________________"
    echo "  All tools executed successfully!"
    echo "___________________________________________"
    echo ""
else
    echo "⚠️ No Python files to process."
fi

exit 0
