let
  pkgs = import <nixpkgs> {};
  unstable = import <nixos-unstable> { allowUnfree = true; config.allowUnfree = true; };
  python = (unstable.python314.withPackages (python-pkgs: [
#      python-pkgs.gymnasium
      python-pkgs.pybox2d
      python-pkgs.numpy
      python-pkgs.matplotlib
      python-pkgs.seaborn
      python-pkgs.pygame
      python-pkgs.tqdm
      python-pkgs.opencv4
#      python-pkgs.tensorboard
#      python-pkgs.tensorflow
      python-pkgs.torchWithRocm
    ]));
in pkgs.mkShell {
  packages = [
	unstable.jetbrains.pycharm
	python
  ];

  shellHook = ''
	export HSA_OVERRIDE_GFX_VERSION=10.3.0
	ln -sfn ${python}/bin/python3 /opt/compilers/current/python
  '';

  NIX_CONFIG_PATH="${toString ./.}/code/nix/config.nix";
}
