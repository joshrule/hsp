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
  };
in
  mach-nix.nixpkgs.mkShell {
    nativeBuildInputs = with mach-nix.nixpkgs; [ autoPatchelfHook ];
    buildInputs = with mach-nix.nixpkgs; [
      pyEnv
      binutils
      util-linux
      stdenv.cc.cc.lib
    ] ;

    # shellHook = '' '';
  }

