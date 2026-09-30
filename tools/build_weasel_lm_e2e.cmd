@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /std:c++17 /EHsc /MT /I D:\Workspace\rime\weasel\include ^
  weasel_lm_e2e.cc /link /LIBPATH:D:\Workspace\rime\weasel\lib64 rime.lib ^
  /OUT:weasel_lm_e2e.exe
if errorlevel 1 exit /b 1
copy /y D:\Workspace\rime\weasel\librime\dist\lib\rime.dll . >nul
copy /y D:\Workspace\rime\third-party\onnxruntime\x64\onnxruntime.dll . >nul
endlocal
