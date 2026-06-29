{name, config}: let
    genPython = with builtins; let
        lib = import <nixpkgs/lib>;
        list = {list = []; __functor = self: item: self//{list=self.list++[item];};};


        pascalCase = str: (lib.strings.toUpper (substring 0 1 str)) + (substring 1 (stringLength str) str);

        genClasses = list: name: set: let
            attrNames = builtins.attrNames set;

            makeMember = memberName: let
                mem=set.${memberName};
            in
                "\t"+memberName+": "+(
                if isAttrs mem then pascalCase memberName
                else if isFloat mem then "float"
                else if isInt mem then "int"
                else if isList mem then "list"
                else if isString mem then "str"
                else if isBool mem then "bool"
                else "Any");

            attrs = map makeMember attrNames;
            className = pascalCase name;
            class = ''
                @dataclass
                class ${className}
                ${builtins.concatStringsSep "\n" attrs}
            '';

            subClasses = el_name: list: let
                el = set.${el_name};
            in
                if builtins.isAttrs el
                    then genClasses list el_name el
                else
                    list;
        in lib.foldr subClasses (list class) attrNames;
    in name: config: builtins.concatStringsSep "\n" (genClasses list name config).list;
in (genPython name config)