with (import <nixpkgs> {});
let
  R-with-packages = pkgs.rWrapper.override{ 
    packages = with pkgs.rPackages; [ 
      stats
      patchwork
      tidyboot
      tidyverse
    ];
  };
in
  mkShell {
    buildInputs = [ R-with-packages ];
  }
