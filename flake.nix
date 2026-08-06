{
  description = "Nix development shell for this repo.";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let overlays = [ (import rust-overlay) ]; in
        let pkgs = import nixpkgs {
          inherit system overlays;
        }; in
        let rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml; in
        {
          devShells.default = with pkgs; mkShell {
            packages = [ rustToolchain rustup cargo-hack just typos ];
          };
        }
      );
}
