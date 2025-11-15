__doc__="
bash ~/code/shitspotter/papers/wacv_2026/scripts/split_pdf.sh ~/code/shitspotter/papers/wacv_2026/main.pdf 10

python ~/code/shitspotter/papers/wacv_2026/scripts/compress_pdf.py main.pdf --quality=screen
"
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $(basename "$0") INPUT_PDF SPLIT_PAGE [OUTPUT_PREFIX]

Split INPUT_PDF into two PDFs at SPLIT_PAGE.
- Part 1: pages 1..SPLIT_PAGE
- Part 2: pages (SPLIT_PAGE+1)..end

OUTPUT_PREFIX (optional) defaults to the input filename (without .pdf).
Examples:
  $(basename "$0") report.pdf 5
  $(basename "$0") report.pdf 12 mysplit
EOF
  exit 1
}

# --- Args & basic checks ---
[[ ${#@} -lt 2 || ${#@} -gt 3 ]] && usage

inpdf=$1
split_page=$2
prefix=${3:-$(basename "${inpdf%.*}")}

[[ ! -f "$inpdf" ]] && { echo "ERROR: '$inpdf' not found."; exit 2; }
[[ ! "$split_page" =~ ^[0-9]+$ ]] && { echo "ERROR: SPLIT_PAGE must be a positive integer."; exit 2; }

out1="${prefix}_part1.pdf"
out2="${prefix}_part2.pdf"

# Detect tools
have_qpdf=false
have_pdftk=false
if command -v qpdf >/dev/null 2>&1; then
  have_qpdf=true
elif command -v pdftk >/dev/null 2>&1; then
  have_pdftk=true
else
  echo "ERROR: Neither 'qpdf' nor 'pdftk' is installed. Please install one and try again."
  exit 3
fi

# Determine total pages (for sanity checks)
total_pages=""
if $have_qpdf; then
  total_pages=$(qpdf --show-npages "$inpdf")
elif command -v pdfinfo >/dev/null 2>&1; then
  # pdfinfo is part of poppler-utils; used only for validation when using pdftk
  total_pages=$(pdfinfo "$inpdf" 2>/dev/null | awk -F': *' '/^Pages:/{print $2}')
fi

if [[ -n "$total_pages" ]]; then
  if (( split_page < 1 )); then
    echo "ERROR: SPLIT_PAGE must be >= 1."
    exit 2
  fi
  if (( split_page >= total_pages )); then
    echo "ERROR: SPLIT_PAGE ($split_page) must be less than total pages ($total_pages)."
    exit 2
  fi
fi

# --- Do the split ---
if $have_qpdf; then
  # qpdf supports 'z' as last page
  qpdf "$inpdf" "$out1" --pages "$inpdf" 1-"$split_page" -- >/dev/null
  next=$(( split_page + 1 ))
  qpdf "$inpdf" "$out2" --pages "$inpdf" "$next"-z -- >/dev/null
else
  # pdftk fallback; uses 'end' keyword
  pdftk A="$inpdf" cat A1-"$split_page" output "$out1"
  next=$(( split_page + 1 ))
  pdftk A="$inpdf" cat A"$next"-end output "$out2"
fi

echo "Done:"
echo "  -> $out1"
echo "  -> $out2"
