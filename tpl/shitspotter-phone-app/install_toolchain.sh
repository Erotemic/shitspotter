#!/usr/bin/env bash
#
# install_toolchain.sh — user-local toolchain installer for the
# shitspotter-phone-app, designed for the shitspotter dev VM.
#
# Default install (always):
#   - Eclipse Temurin JDK 17 (Linux x64)
#   - Android SDK command-line tools
#   - Via sdkmanager: platform-tools, platforms;android-34, build-tools;34.0.0
#   - Gradle is intentionally NOT installed; the gradle wrapper handles that
#     once the project is scaffolded.
#
# Optional add-ons (opt-in via flags):
#   --with-ndk       Android NDK (~3 GB). Needed only for native C/C++ in app.
#   --with-flutter   Flutter SDK (~2 GB), for evaluating the Flutter fallback.
#   --with-rust      Rust + Cargo via rustup (~600 MB), for the Rust+Slint
#                    fallback or a shared Rust core. Linux desktop only by
#                    default — Android targets are added by the script.
#   --all            All of the above.
#
# Other modes:
#   --check          Verify a previous install and exit.
#   --print-env      Print the env-sourcing file and exit (no install).
#   -h | --help      Show this header.
#
# After installing, source the generated env file in any shell that needs it:
#   source /data/tmp/shitspotter-app-toolchain/env.sh
#
# Override install location with TOOLCHAIN_ROOT=/some/path before running.
# Default is /data/tmp/shitspotter-app-toolchain on this VM (2+ TB free), with
# a symlink at $HOME/.local/share/shitspotter-app-toolchain for convenience.
#
# Disk: ~1 GB minimal, ~6 GB --all. The script warns if free space is low.

set -euo pipefail

DEFAULT_ROOT="/data/tmp/shitspotter-app-toolchain"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$DEFAULT_ROOT}"
JDK_DIR="$TOOLCHAIN_ROOT/jdk"
SDK_DIR="$TOOLCHAIN_ROOT/android-sdk"
NDK_DIR_ROOT="$SDK_DIR/ndk"
FLUTTER_DIR="$TOOLCHAIN_ROOT/flutter"
RUST_HOME="$TOOLCHAIN_ROOT/rust"        # CARGO_HOME and RUSTUP_HOME both live under here
DOWNLOAD_DIR="$TOOLCHAIN_ROOT/downloads"
ENV_FILE="$TOOLCHAIN_ROOT/env.sh"
HOME_SYMLINK="$HOME/.local/share/shitspotter-app-toolchain"

# Pin versions so reruns are reproducible.
JDK_VERSION="17.0.12+7"
JDK_TARBALL="OpenJDK17U-jdk_x64_linux_hotspot_17.0.12_7.tar.gz"
JDK_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-${JDK_VERSION}/${JDK_TARBALL}"
JDK_SHA256="9d4dd339bf7e6a9dcba8347661603b74c61ab2a5083ae67bf76da6285da8a778"

CMDLINE_VERSION="11076708_latest"
CMDLINE_ZIP="commandlinetools-linux-${CMDLINE_VERSION}.zip"
CMDLINE_URL="https://dl.google.com/android/repository/${CMDLINE_ZIP}"

ANDROID_PLATFORM="android-34"
ANDROID_BUILD_TOOLS="34.0.0"
ANDROID_NDK_VERSION="26.3.11579264"

# Flutter: pin a recent stable. The version sets the Dart SDK and tooling
# version; the agent can change it if the chosen-stack decision lands on a
# specific Flutter LTS.
FLUTTER_VERSION="3.24.5"
FLUTTER_TARBALL="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TARBALL}"

# Rust: rustup-init bootstraps a minimal toolchain; we then add Android targets.
RUSTUP_INIT_URL="https://sh.rustup.rs"
RUST_ANDROID_TARGETS=(
  aarch64-linux-android
  armv7-linux-androideabi
  x86_64-linux-android
  i686-linux-android
)

WITH_NDK=0
WITH_FLUTTER=0
WITH_RUST=0
DO_INSTALL=1
DO_CHECK=0
PRINT_ENV_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --with-ndk)     WITH_NDK=1 ;;
    --with-flutter) WITH_FLUTTER=1 ;;
    --with-rust)    WITH_RUST=1 ;;
    --all)          WITH_NDK=1; WITH_FLUTTER=1; WITH_RUST=1 ;;
    --check)        DO_INSTALL=0; DO_CHECK=1 ;;
    --print-env)    DO_INSTALL=0; PRINT_ENV_ONLY=1 ;;
    -h|--help)      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[install_toolchain] %s\n' "$*" >&2; }

check_disk() {
  local avail_kb
  avail_kb=$(df -Pk "$TOOLCHAIN_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
  if [ -z "$avail_kb" ]; then
    avail_kb=$(df -Pk "$(dirname "$TOOLCHAIN_ROOT")" | awk 'NR==2 {print $4}')
  fi
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  log "free space at $TOOLCHAIN_ROOT: ${avail_gb} GiB"
  if [ "$avail_gb" -lt 5 ]; then
    log "WARNING: less than 5 GiB free — install may fail partway through"
  fi
}

ensure_home_symlink() {
  # Convenience: predictable path under $HOME pointing at the real install.
  if [ "$TOOLCHAIN_ROOT" = "$HOME_SYMLINK" ]; then return; fi
  mkdir -p "$(dirname "$HOME_SYMLINK")"
  if [ -L "$HOME_SYMLINK" ]; then
    local cur; cur=$(readlink "$HOME_SYMLINK")
    if [ "$cur" != "$TOOLCHAIN_ROOT" ]; then
      log "updating $HOME_SYMLINK -> $TOOLCHAIN_ROOT (was $cur)"
      ln -sfn "$TOOLCHAIN_ROOT" "$HOME_SYMLINK"
    fi
  elif [ -e "$HOME_SYMLINK" ]; then
    log "WARNING: $HOME_SYMLINK exists and is not a symlink — leaving alone"
  else
    ln -s "$TOOLCHAIN_ROOT" "$HOME_SYMLINK"
    log "linked $HOME_SYMLINK -> $TOOLCHAIN_ROOT"
  fi
}

download() {
  # download URL DEST [SHA256]
  local url="$1" dest="$2" sha="${3:-}"
  if [ -f "$dest" ]; then
    log "already downloaded: $(basename "$dest")"
  else
    log "downloading $(basename "$dest")"
    curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
    if [ -n "$sha" ]; then
      if ! echo "$sha  $dest.part" | sha256sum -c - >/dev/null; then
        log "sha256 mismatch for $(basename "$dest") — leaving .part for inspection"
        return 1
      fi
    fi
    mv "$dest.part" "$dest"
  fi
  if [ -n "$sha" ]; then
    echo "$sha  $dest" | sha256sum -c - >/dev/null
  fi
}

install_jdk() {
  if [ -x "$JDK_DIR/bin/java" ]; then
    local v
    v=$("$JDK_DIR/bin/java" -version 2>&1 | head -1)
    log "JDK already present: $v"
    return
  fi
  download "$JDK_URL" "$DOWNLOAD_DIR/$JDK_TARBALL" "$JDK_SHA256"
  log "extracting JDK to $JDK_DIR"
  rm -rf "$JDK_DIR.tmp"
  mkdir -p "$JDK_DIR.tmp"
  tar -xzf "$DOWNLOAD_DIR/$JDK_TARBALL" -C "$JDK_DIR.tmp" --strip-components=1
  mv "$JDK_DIR.tmp" "$JDK_DIR"
}

install_cmdline_tools() {
  local target="$SDK_DIR/cmdline-tools/latest"
  if [ -x "$target/bin/sdkmanager" ]; then
    log "Android cmdline-tools already present"
    return
  fi
  download "$CMDLINE_URL" "$DOWNLOAD_DIR/$CMDLINE_ZIP"
  log "extracting cmdline-tools to $target"
  rm -rf "$SDK_DIR/cmdline-tools/_extract" "$target"
  mkdir -p "$SDK_DIR/cmdline-tools/_extract"
  unzip -q "$DOWNLOAD_DIR/$CMDLINE_ZIP" -d "$SDK_DIR/cmdline-tools/_extract"
  # The zip contains a top-level "cmdline-tools" dir; rename it to "latest"
  # which is the layout sdkmanager expects.
  mkdir -p "$SDK_DIR/cmdline-tools"
  mv "$SDK_DIR/cmdline-tools/_extract/cmdline-tools" "$target"
  rm -rf "$SDK_DIR/cmdline-tools/_extract"
}

run_sdkmanager() {
  JAVA_HOME="$JDK_DIR" PATH="$JDK_DIR/bin:$PATH" \
    "$SDK_DIR/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$SDK_DIR" "$@"
}

install_sdk_packages() {
  local pkgs=(
    "platform-tools"
    "platforms;${ANDROID_PLATFORM}"
    "build-tools;${ANDROID_BUILD_TOOLS}"
  )
  if [ "$WITH_NDK" -eq 1 ]; then
    pkgs+=("ndk;${ANDROID_NDK_VERSION}")
  fi
  log "accepting SDK licenses"
  yes | run_sdkmanager --licenses >/dev/null || true
  log "installing SDK packages: ${pkgs[*]}"
  run_sdkmanager --install "${pkgs[@]}"
}

install_flutter() {
  if [ -x "$FLUTTER_DIR/bin/flutter" ]; then
    local v
    v=$("$FLUTTER_DIR/bin/flutter" --version 2>&1 | head -1 || echo unknown)
    log "Flutter already present: $v"
    return
  fi
  download "$FLUTTER_URL" "$DOWNLOAD_DIR/$FLUTTER_TARBALL"
  log "extracting Flutter to $FLUTTER_DIR"
  rm -rf "$FLUTTER_DIR.tmp" "$FLUTTER_DIR"
  mkdir -p "$FLUTTER_DIR.tmp"
  tar -xJf "$DOWNLOAD_DIR/$FLUTTER_TARBALL" -C "$FLUTTER_DIR.tmp" --strip-components=1
  mv "$FLUTTER_DIR.tmp" "$FLUTTER_DIR"
  log "disabling Flutter analytics (non-blocking)"
  "$FLUTTER_DIR/bin/flutter" --disable-analytics >/dev/null 2>&1 || true
  log "pre-caching Flutter desktop+android artifacts (this can take a few minutes)"
  PATH="$JDK_DIR/bin:$FLUTTER_DIR/bin:$PATH" \
    JAVA_HOME="$JDK_DIR" \
    "$FLUTTER_DIR/bin/flutter" precache --linux --android >/dev/null 2>&1 || \
      log "flutter precache returned non-zero (will be re-tried lazily on first build)"
}

install_rust() {
  if [ -x "$RUST_HOME/cargo/bin/cargo" ]; then
    local v
    v=$(CARGO_HOME="$RUST_HOME/cargo" RUSTUP_HOME="$RUST_HOME/rustup" \
        "$RUST_HOME/cargo/bin/cargo" --version 2>&1 || echo unknown)
    log "Rust already present: $v"
  else
    download "$RUSTUP_INIT_URL" "$DOWNLOAD_DIR/rustup-init.sh"
    log "running rustup-init (toolchain → $RUST_HOME)"
    chmod +x "$DOWNLOAD_DIR/rustup-init.sh"
    CARGO_HOME="$RUST_HOME/cargo" RUSTUP_HOME="$RUST_HOME/rustup" \
      "$DOWNLOAD_DIR/rustup-init.sh" -y --no-modify-path \
        --default-toolchain stable --profile minimal >/dev/null
  fi
  log "ensuring Android Rust targets are installed"
  CARGO_HOME="$RUST_HOME/cargo" RUSTUP_HOME="$RUST_HOME/rustup" \
    "$RUST_HOME/cargo/bin/rustup" target add "${RUST_ANDROID_TARGETS[@]}" >/dev/null
}

write_env_file() {
  {
    cat <<EOF
# Auto-generated by install_toolchain.sh — source this file to use the
# shitspotter-phone-app toolchain in your current shell.
export SHITSPOTTER_APP_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT"
export JAVA_HOME="$JDK_DIR"
export ANDROID_HOME="$SDK_DIR"
export ANDROID_SDK_ROOT="$SDK_DIR"
EOF
    if [ "$WITH_NDK" -eq 1 ] || [ -d "$NDK_DIR_ROOT/$ANDROID_NDK_VERSION" ]; then
      echo "export ANDROID_NDK_HOME=\"$NDK_DIR_ROOT/$ANDROID_NDK_VERSION\""
    fi
    if [ "$WITH_FLUTTER" -eq 1 ] || [ -x "$FLUTTER_DIR/bin/flutter" ]; then
      echo "export FLUTTER_HOME=\"$FLUTTER_DIR\""
    fi
    if [ "$WITH_RUST" -eq 1 ] || [ -x "$RUST_HOME/cargo/bin/cargo" ]; then
      cat <<EOF
export CARGO_HOME="$RUST_HOME/cargo"
export RUSTUP_HOME="$RUST_HOME/rustup"
EOF
    fi
    cat <<EOF
_ssp_path="\$JAVA_HOME/bin:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$ANDROID_HOME/build-tools/${ANDROID_BUILD_TOOLS}"
EOF
    if [ "$WITH_FLUTTER" -eq 1 ] || [ -x "$FLUTTER_DIR/bin/flutter" ]; then
      echo '_ssp_path="$_ssp_path:$FLUTTER_HOME/bin"'
    fi
    if [ "$WITH_RUST" -eq 1 ] || [ -x "$RUST_HOME/cargo/bin/cargo" ]; then
      echo '_ssp_path="$_ssp_path:$CARGO_HOME/bin"'
    fi
    cat <<'EOF'
export PATH="$_ssp_path:$PATH"
unset _ssp_path
EOF
  } > "$ENV_FILE"
  log "wrote env file: $ENV_FILE"
}

do_check() {
  local fail=0
  for f in \
    "$JDK_DIR/bin/java" \
    "$SDK_DIR/cmdline-tools/latest/bin/sdkmanager" \
    "$SDK_DIR/platform-tools/adb" \
    "$SDK_DIR/platforms/${ANDROID_PLATFORM}/android.jar" \
    "$SDK_DIR/build-tools/${ANDROID_BUILD_TOOLS}/aapt2" \
    "$ENV_FILE" \
  ; do
    if [ -e "$f" ]; then printf '  ok   %s\n' "$f"
    else printf '  MISS %s\n' "$f"; fail=1; fi
  done
  if [ "$WITH_NDK" -eq 1 ]; then
    if [ -d "$NDK_DIR_ROOT/$ANDROID_NDK_VERSION" ]; then
      printf '  ok   ndk/%s\n' "$ANDROID_NDK_VERSION"
    else printf '  MISS ndk/%s\n' "$ANDROID_NDK_VERSION"; fail=1; fi
  fi
  if [ "$WITH_FLUTTER" -eq 1 ]; then
    if [ -x "$FLUTTER_DIR/bin/flutter" ]; then printf '  ok   flutter\n'
    else printf '  MISS flutter\n'; fail=1; fi
  fi
  if [ "$WITH_RUST" -eq 1 ]; then
    if [ -x "$RUST_HOME/cargo/bin/cargo" ]; then printf '  ok   rust\n'
    else printf '  MISS rust\n'; fail=1; fi
  fi
  return $fail
}

if [ "$PRINT_ENV_ONLY" -eq 1 ]; then
  if [ ! -f "$ENV_FILE" ]; then
    echo "no env file at $ENV_FILE — run install first" >&2; exit 1
  fi
  cat "$ENV_FILE"; exit 0
fi

if [ "$DO_CHECK" -eq 1 ]; then
  do_check && log "toolchain looks complete" || { log "toolchain incomplete"; exit 1; }
  exit 0
fi

mkdir -p "$TOOLCHAIN_ROOT" "$DOWNLOAD_DIR" "$SDK_DIR"
ensure_home_symlink
check_disk
install_jdk
install_cmdline_tools
install_sdk_packages
[ "$WITH_FLUTTER" -eq 1 ] && install_flutter
[ "$WITH_RUST"    -eq 1 ] && install_rust
write_env_file

log "done. To use the toolchain in this shell, run:"
log "  source $ENV_FILE"
log "Then verify with:"
log "  java -version && adb --version && sdkmanager --version"
[ "$WITH_FLUTTER" -eq 1 ] && log "  flutter --version"
[ "$WITH_RUST"    -eq 1 ] && log "  cargo --version && rustup target list --installed"
true
