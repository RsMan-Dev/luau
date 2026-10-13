{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [ cmake ninja pkg-config vulkan-headers vulkan-loader shaderc spirv-tools spirv-cross glslang SDL2 python3 ];
}
