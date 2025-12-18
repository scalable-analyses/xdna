{
  description = "Dev environment for Hello XDNA!";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];

      forEachSystem = f:
        nixpkgs.lib.genAttrs systems (system:
          let
            pkgs = import nixpkgs { inherit system; };
          in
            f pkgs
        );
    in {
      # One dev shell per platform, named "default"
      devShells = forEachSystem (pkgs: {
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              python313
              (python313.withPackages (ps: with ps; [
                sphinx-autobuild
                sphinx-book-theme
                sphinx-copybutton
              ]))
              uv
            ];
          };
        });
    };
}
