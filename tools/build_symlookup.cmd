@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /std:c++17 /EHsc /MT symlookup.cpp /link dbghelp.lib /OUT:symlookup.exe
