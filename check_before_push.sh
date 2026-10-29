#!/usr/bin/env bash
# Run from the repo root before every `git push`:  bash check_before_push.sh
# Flags anything that should not go to a public GitHub repo. Read-only.
cd "${1:-.}" || exit 1
fail=0
section() { printf '\n== %s\n' "$1"; }
report() {
  local hits; hits=$(printf '%s\n' "$2" | sed '/^[[:space:]]*$/d')
  if [ -n "$hits" ]; then echo "$hits"; fail=1; else echo "ok"; fi
}

FILES=$(find . -type f -not -path './.git/*' -not -name 'check_before_push.sh')

section "Files over 50 MB (GitHub rejects >100 MB; put these on Zenodo)"
report big "$(find . -type f -size +50M -not -path './.git/*')"

section "Trajectory / checkpoint files (belong on Zenodo)"
report traj "$(echo "$FILES" | grep -Ei '\.(xtc|trr|dcd|nc|dtr|cpt|edr|h5)$|_trj/')"

section "Secrets, keys and license files"
report sec "$(echo "$FILES" | grep -Ei '\.env$|\.lic$|\.pem$|id_rsa|credentials')
$(echo "$FILES" | xargs grep -IlE 'gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE|LICENSE_KEY|MODELLER_LICENSE|api[_-]?key *=' 2>/dev/null)"

section "Hard-coded personal or cluster paths (replace with relative paths)"
report paths "$(echo "$FILES" | xargs grep -InE '/Users/[A-Za-z]|/bigdata/|/rhome/|/home/[a-z]' 2>/dev/null | cut -c1-160)"

section "Working notes and handoff files"
report agent "$(echo "$FILES" | grep -Ei 'AGENTS\.md|PROJECT_(CONTEXT|MEMORY)\.md|HANDOFF|handoff|/\.codex/|\.sha256$|audit')"

section "Third-party code that may not be redistributable (link to it instead)"
report vendor "$(echo "$FILES" | grep -Ei 'vendor|d2D-version|delta2d.*/(ORIGINAL|bin)/|sparta' )"

section "Clutter (pycache, .DS_Store, Office files, node_modules)"
report clutter "$(echo "$FILES" | grep -Ei '__pycache__|\.pyc$|\.DS_Store|\.docx?$|\.xlsx$|\.pptx$|~\$|node_modules|\.ipynb_checkpoints')"

printf '\n'
if [ $fail -eq 0 ]; then echo "All checks passed."; else echo "Review the items above before pushing."; fi
exit $fail
