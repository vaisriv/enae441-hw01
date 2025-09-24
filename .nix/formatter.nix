{
    pkgs,
    inputs,
    ...
}:
inputs.treefmt-nix.lib.mkWrapper pkgs {
    projectRootFile = "flake.nix";

    # nix fmt
    programs = {
        deadnix.enable = true;
        alejandra.enable = true;
    };
    settings.formatter = {
        alejandra = {
            options = [
                "--experimental-config"
                "./.nix/fmt/alejandra.toml"
            ];
        };
    };

    # python
    programs = {
        ruff-check.enable = true;
        ruff-format.enable = true;
    };

    # latex
    programs = {
        latexindent.enable = true;
    };
}
