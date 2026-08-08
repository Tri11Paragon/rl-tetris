config: let
    genRust =
    let
        lib = import <nixpkgs/lib>;
        list = {list = []; __functor = self: item: self//{list=self.list++[item];};};
        pascalCase = str: (lib.strings.toUpper (builtins.substring 0 1 str)) + (builtins.substring 1 (builtins.stringLength str) str);

      rustType = memberName: value:
        if builtins.isAttrs value then pascalCase memberName
        else if builtins.isFloat value then "f64"
        else if builtins.isInt value then "i64"
        else if builtins.isString value then "String"
        else if builtins.isBool value then "bool"
        else if builtins.isList value then rustListType memberName value
        else "serde_json::Value";

      rustListType = memberName: value:
        if value == [] then
          "Vec<serde_json::Value>"
        else
          let
            first = builtins.elemAt value 0;
          in
            if builtins.isAttrs first then "Vec<" + pascalCase memberName + "Item>"
            else if builtins.isFloat first then "Vec<f64>"
            else if builtins.isInt first then "Vec<i64>"
            else if builtins.isString first then "Vec<String>"
            else if builtins.isBool first then "Vec<bool>"
            else if builtins.isList first then "Vec<serde_json::Value>"
            else "Vec<serde_json::Value>";

      makeRustMember = set: memberName:
        let
          mem = set.${memberName};
        in
          "    pub " + memberName + ": " + rustType memberName mem + ",";

      makeRustStruct = name: set:
        let
          attrNames = builtins.attrNames set;
          structName = pascalCase name;
          fields = map (makeRustMember set) attrNames;
        in
          ''
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            pub struct ${structName} {
            ${builtins.concatStringsSep "\n" fields}
            }
          '';

      genStructs = acc: name: set:
        let
          attrNames = builtins.attrNames set;

          currentStruct = makeRustStruct name set;

          subStructs = elName: acc:
            let
              el = set.${elName};
            in
              if builtins.isAttrs el then
                genStructs acc elName el
              else if builtins.isList el && el != [] && builtins.isAttrs (builtins.elemAt el 0) then
                genStructs acc "${elName}Item" (builtins.elemAt el 0)
              else
                acc;
        in
          lib.foldr subStructs (acc currentStruct) attrNames;
    in
      name: config:
        ''
          use serde::{Deserialize, Serialize};

          ${builtins.concatStringsSep "\n" (genStructs list name config).list}
        '';
    name = "NNConfig";
in (genRust name config)