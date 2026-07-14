{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    gnumake
    pkg-config
    shaderc
    vulkan-headers
    vulkan-loader
    vulkan-tools
    vulkan-validation-layers
  ];

  shellHook = ''
    export VULKAN_SDK="${pkgs.vulkan-headers}"
    export CPATH="${pkgs.vulkan-headers}/include:${pkgs.vulkan-loader}/include:$CPATH"
    export LIBRARY_PATH="${pkgs.vulkan-loader}/lib:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"
  '';
}
