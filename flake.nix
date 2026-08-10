{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    engine = {
        url = "path:./engine";
        inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, engine }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      makePython = torchPackage:
        pkgs.python314.withPackages (pythonPackages: [
          engine.packages.${system}.default
          pythonPackages.pybox2d
          pythonPackages.numpy
          pythonPackages.matplotlib
          pythonPackages.seaborn
          pythonPackages.pygame
          pythonPackages.tqdm
          pythonPackages.opencv4
          pythonPackages.openai
          pythonPackages.ollama
          pythonPackages.requests
          (torchPackage pythonPackages)
        ]);

      makeShell = {
        python,
        rocm ? false,
      }:
        pkgs.mkShell {
          packages = [
            python
            pkgs.cargo
            pkgs.rustc
            pkgs.maturin
          ];

          shellHook = ''
            ${if rocm then ''
              export HSA_OVERRIDE_GFX_VERSION=10.3.0
            '' else ""}
            export NIX_CONFIG_PATH="$PWD/code/nix/config.nix"
            ln -sfn ${python}/bin/python3 /opt/compilers/current/python

            echo "Python: $(python --version)"
            zsh
          '';
        };
    in
    {
      devShells.${system} = {
        default = makeShell {
          python = makePython (pypk: pypk.torch);
        };

        rocm = makeShell {
          python = makePython (pypk: pypk.torchWithRocm);
          rocm = true;
        };

        cuda = makeShell {
          python = makePython (pypk: pypk.torchWithCuda);
        };

        cuda1060 = makeShell {
          python = makePython (pypk: (pypk.torch.override {
            gpuTargets = ["sm_61"];
            cudaSupport = true;
          }));
        };
      };
    };
}