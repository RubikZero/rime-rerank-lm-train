@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /std:c++17 /EHsc /MT /utf-8 /D_UNICODE /DUNICODE ^
  /I D:\Workspace\rime\weasel\include ^
  /I D:\Workspace\rime\third-party\boost_1_84_0 ^
  weasel_ipc_maintenance_e2e.cc /link ^
  /LIBPATH:D:\Workspace\rime\weasel\msbuild\Release\x64 ^
  /LIBPATH:D:\Workspace\rime\third-party\boost_1_84_0\stage\lib ^
  WeaselIPC.lib kernel32.lib user32.lib advapi32.lib shell32.lib ole32.lib ^
  /OUT:weasel_ipc_maintenance_e2e.exe
endlocal
