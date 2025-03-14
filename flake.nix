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

            packages = [
              pkgs.python312Packages.python-lsp-server
              git gitRepo gnupg autoconf curl
              procps gnumake util-linux m4 gperf unzip
              cudatoolkit linuxPackages.nvidia_x11
              libGLU libGL
              xorg.libXi xorg.libXmu freeglut
              xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
              ncurses5 stdenv.cc binutils
            ]; # pkgs.python3.12-python-lsp-server ?

            languages.python = {
              enable = true;
              package = pkgs.python312;
              venv.enable = true;
              venv.requirements = (builtins.readFile ./requirements.txt);
            };
            shellHook = ''
              export CUDA_PATH=${pkgs.cudatoolkit}
              # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
            '';  
          })
          ];
        };
      }
    );
}
