@echo off
rem Fetch the librime-lua plugin (and its bundled Lua sources) into the
rem librime plugin directory.  The librime CMake build auto-discovers every
rem subdirectory of plugins/ and merges the plugin into rime.dll.
setlocal
set LUA_DIR=%~dp0..\..\..\weasel\librime\plugins\lua
if not exist "%LUA_DIR%" (
  echo cloning librime-lua ...
  git clone --depth 1 https://github.com/hchunhui/librime-lua.git "%LUA_DIR%"
  if errorlevel 1 exit /b 1
)
if not exist "%LUA_DIR%\thirdparty\lua5.4\lua.h" (
  echo fetching bundled Lua sources ...
  pushd "%LUA_DIR%"
  call action-install.bat
  if errorlevel 1 (popd & exit /b 1)
  popd
)
echo lua plugin ready: %LUA_DIR%
exit /b 0
