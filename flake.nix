{
  description = "Machine Learning - Team E";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/24.11";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, flake-utils, devenv }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            ({ pkgs, config, ... }: {

            packages = [ pkgs.python312Packages.python-lsp-server]; # pkgs.python3.12-python-lsp-server ?

            languages.python = {
              enable = true;
              package = pkgs.python312;
              venv.enable = true;
              venv.requirements = (builtins.readFile ./requirements.txt);
            };
          })
          ];
        };
      }
    );
}
