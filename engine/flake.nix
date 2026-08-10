{
  description = "engine Python package built from Rust with maturin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f import nixpkgs {
            inherit system;
          }
        );
    in
    {
      packages = forAllSystems (
        pkgs:
        {
          default = pkgs.python311Packages.buildPythonPackage rec {
            pname = "engine";
            version = "0.1.0";

            pyproject = true;

            src = pkgs.lib.cleanSource ./.;

            nativeBuildInputs = [
              pkgs.rustPlatform.cargoSetupHook
              pkgs.rustPlatform.maturinBuildHook
              pkgs.pkg-config
            ];

            buildInputs = [
              pkgs.python311
            ];

            propagatedBuildInputs = [
              pkgs.python311Packages.numpy
            ];

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
            };

            pythonImportsCheck = [
              "engine"
            ];
          };
        }
      );

      devShells = forAllSystems (
        pkgs:
        {
          default =
            let
              engine = self.packages.${pkgs.system}.default;
              python = pkgs.python311.withPackages (
                ps: [
                  engine
                  ps.numpy
                  ps.maturin
                ]
              );
            in
            pkgs.mkShell {
              packages = [
                python
                pkgs.cargo
                pkgs.rustc
                pkgs.maturin
              ];
            };
        }
      );
    };
}