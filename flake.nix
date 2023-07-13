{
  description = "R flake";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs, ...}:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
      R-with-packages = forAllSystems (system: pkgs.${system}.rWrapper.override { 
        packages = with pkgs.${system}.rPackages; [ 
          patchwork
          tidyboot
          tidyverse
        ];
      });
    in
      {
        devShells = forAllSystems (system: {
          default = pkgs.${system}.mkShell {
            buildInputs = with pkgs.${system}; [
              R-with-packages.${system}
            ];
          };
        });
      };
}
