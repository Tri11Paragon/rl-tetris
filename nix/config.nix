{ file }: let
    lib = import <nixpkgs/lib>;
    path = file;

    math = rec {
        pow = base: exponent: if exponent < 0 then 1.0/(pow base (-exponent)) else if exponent == 0 then 1.0 else (pow base (exponent - 1)) * base;
    };

    helpers = rec {
        e = math.pow 10;
        e- = x: e (-x);
        sequential = x: "nn.Sequential(${x})";
    };

    default = import ./defaults.nix helpers;
    config = if builtins.pathExists path then import path helpers else {};

    updated_attribs = lib.recursiveUpdate default config;

    python_types = import ./gentypes.nix updated_attribs;

in {
    config_data=updated_attribs;
    python=python_types;
}