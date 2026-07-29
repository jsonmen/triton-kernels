{
  description = "Unified development environment for PyTorch, Triton, and uv package manager";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        myPython = pkgs.python313;

        # 1. Provide the ldconfig shim for Triton/CUDA validation
        ldconfig-shim = pkgs.writeShellScriptBin "ldconfig" ''
          if [ "$1" = "-p" ]; then
            echo "libcuda.so.1 (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so.1"
            exit 0
          fi
          exec ${pkgs.glibc.bin}/bin/ldconfig "$@"
        '';

        # Combined dependencies from both files
        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glibc
          libffi
          openssl
          libxcrypt-legacy
          linuxPackages.nvidia_x11
          cudaPackages.cuda_cudart
          cudaPackages.libcublas
          libglvnd
          libGL
          glib

          libxcb
          libX11
          libXext
          libXrender
          libICE
          libSM
        ];

        libPath = pkgs.lib.makeLibraryPath libs;
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            ldconfig-shim
            uv
            which
            black
            myPython
            gcc # Crucial: Triton calls 'gcc' to build CUDA utils
            cudaPackages.cuda_nvcc
            pkg-config
            coreutils
          ];

          shellHook = ''
            # --- Fix Direnv Nesting ---
            if [ -n "$IN_NIX_SHELL_ACTIVATED" ]; then exit 0; fi
            export IN_NIX_SHELL_ACTIVATED=1

            # --- Setup Paths (Including /run/opengl-driver/lib for host drivers) ---
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${libPath}:''${LD_LIBRARY_PATH:-}"
            export CUDA_PATH="${pkgs.linuxPackages.nvidia_x11}"
            export TRITON_PTXAS_PATH="${pkgs.cudaPackages.cuda_nvcc}/bin/ptxas"

            # --- Triton Fix for NixOS/Nix ---
            # Bypasses the hardcoded '/sbin/ldconfig' lookup
            export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"

            # --- Nix-LD (for UV unpatched binaries) ---
            export NIX_LD_LIBRARY_PATH="${libPath}"
            export NIX_LD="''$(cat ${pkgs.stdenv.cc}/nix-support/dynamic-linker)"

            # --- Python/Triton specific configuration ---
            export UV_PYTHON="${myPython}/bin/python3"
            export UV_PYTHON_DOWNLOADS="never"
            export CPATH="${myPython}/include/python3.11:''${CPATH:-}"

            echo "=== PyTorch & Triton Environment ==="

            if [ ! -d ".venv" ]; then
              echo "Initializing local .venv using Nix Python..."
              uv venv --python "${myPython}/bin/python3"
            fi

            source .venv/bin/activate

            if [ -z "$DIRENV_IN_ENVRC" ]; then
              echo "🚀 Environment optimized for Triton/PyTorch Inductor."
            fi
          '';
        };
      }
    );
}
