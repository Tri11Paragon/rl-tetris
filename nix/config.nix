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

    validateKeys = prefix: defaults: supplied:
        builtins.foldl'
            (checked: key:
                let
                    keyPath =
                        if prefix == ""
                        then key
                        else "${prefix}.${key}";
                in
                    if !builtins.hasAttr key defaults then
                        throw "Unknown configuration key: ${keyPath}"
                    else
                        let
                            defaultValue = defaults.${key};
                            suppliedValue = supplied.${key};
                            nestedCheck =
                                if builtins.isAttrs defaultValue
                                    && builtins.isAttrs suppliedValue
                                then validateKeys keyPath defaultValue suppliedValue
                                else null;
                        in builtins.seq nestedCheck checked)
            null
            (builtins.attrNames supplied);

    validatedConfig =
        builtins.seq (validateKeys "" default config) config;

    updated_attribs = lib.recursiveUpdate default validatedConfig;

    python_types = import ./gentypes.nix updated_attribs;
    rust_types = import ./genrustypes.nix updated_attribs;

in {
    config_data=updated_attribs;
    python=python_types;
    rust=rust_types;
}