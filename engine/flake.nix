{
  description = "engine Python package built from Rust with maturin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
      };
    in
    {
      package = pkgs: python:
        let
            python_pkgs = python.pkgs;
        in python_pkgs.buildPythonPackage rec {
            pname = "tetris";
            version = "0.1.0";

            pyproject = true;

            src = pkgs.lib.cleanSource ./.;

            nativeBuildInputs = [
              pkgs.rustPlatform.cargoSetupHook
#              pkgs.rustPlatform.maturinBuildHook
              pkgs.cargo
              pkgs.maturin
              pkgs.rustc
              pkgs.pkg-config
            ];

            buildInputs = [
              python
            ];

            propagatedBuildInputs = [
              python_pkgs.numpy
            ];

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
            };

            pythonImportsCheck = [
              "tetris"
            ];
        };
    };
}