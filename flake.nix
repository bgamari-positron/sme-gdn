{
  description = "Gated DeltaNet AArch64 SME kernel checks";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/7a0f122f5090cf4c2ade2a13a0e229d4e19ba71f";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
    in {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          crossCompiler = pkgs.pkgsCross.aarch64-multiplatform.pkgsStatic.stdenv.cc;
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.gnumake
              crossCompiler
              (pkgs.qemu.override { userOnly = true; })
            ];
            CROSS_CXX = "${crossCompiler}/bin/${crossCompiler.targetPrefix}c++";
          };
        });
    };
}
