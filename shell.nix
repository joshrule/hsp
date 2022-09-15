let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {
    python = "python39";
    pypiDataRev = "c1c1684ceb2a2db22d76111922e6e81ff7b3111b";
    pypiDataSha256 = "1kzh5v779c9gm16fcb1rvrm5iqnnyk5dq32z8jjmxpglasdvsjlg";
  };
  pyEnv = mach-nix.mkPython rec {
    requirements = ''
      pandas
      pygame
      numpy
      progressbar2
      torch
      visdom
      gym[classic_control]
    '';
    providers.pygame = "nixpkgs";
    packagesExtra = [
      # https://github.com/openai/gym/archive/refs/tags/0.25.2.tar.gz
      https://github.com/tesatory/mazebase/archive/refs/tags/v0.1.tar.gz
    ];
  };
in
  mach-nix.nixpkgs.mkShell {
    nativeBuildInputs = with mach-nix.nixpkgs; [ autoPatchelfHook ];
    # LD_LIBRARY_PATH="${mach-nix.nixpkgs.xorg.libX11}/lib:${mach-nix.nixpkgs.libGL}/lib:${mach-nix.nixpkgs.libGLU}/lib:${mach-nix.nixpkgs.freeglut}/lib:${mach-nix.nixpkgs.stdenv.cc.cc.lib}/lib";
    buildInputs = with mach-nix.nixpkgs; [
      pyEnv
      binutils
      util-linux
      stdenv.cc.cc.lib
      # mesa
      # libGL
      # libGLU
      # xorg.libX11
      # freeglut
    ] ;

    # shellHook = '' '';
  }

