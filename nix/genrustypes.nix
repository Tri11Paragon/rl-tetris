config: let
  genRust =
    let
      lib = import <nixpkgs/lib>;

      list = {
        list = [];
        __functor = self: item: self // {
          list = self.list ++ [item];
        };
      };

      chars = builtins.stringToCharacters;

      isUpper = ch: ch == lib.strings.toUpper ch && ch != lib.strings.toLower ch;
      isLower = ch: ch == lib.strings.toLower ch && ch != lib.strings.toUpper ch;

      capitalize = str:
        if str == "" then ""
        else
          (lib.strings.toUpper (builtins.substring 0 1 str))
          + (builtins.substring 1 (builtins.stringLength str) str);

      splitOnUnderscore = str:
        builtins.filter (part: part != "") (lib.splitString "_" str);

      /*
        Converts:
          init_lr -> InitLr
          min_lr -> MinLr
          batchSize -> BatchSize
          NNConfig -> NNConfig
          actionsRewardItem -> ActionsRewardItem
      */
      pascalCase = str:
        let
          parts = splitOnUnderscore str;
        in
          builtins.concatStringsSep "" (map capitalize parts);

      /*
        Convert mixed camelCase/PascalCase/snake_case-ish names to Rust snake_case.

        Examples:
          batchSize -> batch_size
          saveInterval -> save_interval
          kl_cutoff -> kl_cutoff
          init_lr -> init_lr
          actorLearnRate -> actor_learn_rate
          NNConfig -> nnconfig with this simple version

        Note:
          If you care about acronyms becoming nn_config instead of nnconfig,
          that needs a more complex acronym-aware converter.
      */
      snakeCase =
        str:
          let
            len = builtins.stringLength str;

            go = i: acc:
              if i >= len then
                acc
              else
                let
                  ch = builtins.substring i 1 str;
                  prev =
                    if i == 0 then ""
                    else builtins.substring (i - 1) 1 str;
                  next =
                    if i + 1 >= len then ""
                    else builtins.substring (i + 1) 1 str;

                  needsUnderscore =
                    i > 0
                    && ch != "_"
                    && prev != "_"
                    && isUpper ch
                    && (
                      isLower prev
                      || (
                        isUpper prev
                        && next != ""
                        && isLower next
                      )
                    );

                  prefix = if needsUnderscore then "_" else "";
                in
                  go (i + 1) (acc + prefix + lib.strings.toLower ch);
          in
            go 0 "";

      rustType = memberName: value:
        if builtins.isAttrs value then pascalCase memberName
        else if builtins.isFloat value then "serde_json::Number"
        else if builtins.isInt value then "serde_json::Number"
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
            else if builtins.isFloat first then "Vec<serde_json::Number>"
            else if builtins.isInt first then "Vec<serde_json::Number>"
#            else if builtins.isString first then "Vec<String>"
#            else if builtins.isBool first then "Vec<bool>"
#            else if builtins.isList first then "Vec<serde_json::Value>"
            else "Vec<serde_json::Value>";

      makeRustMember = set: memberName:
        let
          mem = set.${memberName};
          rustMemberName = snakeCase memberName;
          serdeRename = memberDef:
            if memberName == rustMemberName then
              "${memberDef}"
            else
              "\n${tab}#[serde(rename = \"${memberName}\")]\n${memberDef}";
          tab = "\t";
        in
          serdeRename "${tab}pub ${rustMemberName}: ${rustType memberName mem},";

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
          ${builtins.concatStringsSep "\n" (genStructs list name config).list}
        '';

  name = "NNConfig";
in
  genRust name config