Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor zlib
SetDatablockOptimize on

!include "MUI2.nsh"
!include "x64.nsh"
!include "nsDialogs.nsh"
!include "Sections.nsh"
!include "FileFunc.nsh"

Var RuntimeMode
Var ModelMode
Var Device
Var DeviceDialog
Var CpuRadio
Var CudaRadio
Var Arguments
Var Option

Name "Voice-Extractor1.0"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\Voice-Extractor"
InstallDirRegKey HKCU "Software\Voice-Extractor" "InstallDir"
BrandingText "Voice Extractor"
InstType "标准安装"
InstType /CUSTOMSTRING=自定义安装
ShowInstDetails show
VIProductVersion "1.0.0.0"
VIAddVersionKey /LANG=2052 "ProductName" "Voice-Extractor1.0"
VIAddVersionKey /LANG=2052 "FileDescription" "Voice Extractor 安装程序"
VIAddVersionKey /LANG=2052 "FileVersion" "1.0.0.0"
VIAddVersionKey /LANG=2052 "LegalCopyright" "Voice Extractor contributors"

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TEXT "安装向导将完成程序、Python、计算依赖和模型的安装。$\r$\n$\r$\n您可以选择安装位置、安装组件、CPU 或 NVIDIA GPU 环境，以及快捷方式。缺失文件会在安装过程中下载，请保持联网。$\r$\n$\r$\n已有强化模型与提取结果会保留。"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${PROJECT_ROOT}\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
Page custom DevicePage DevicePageLeave
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE DirectoryPageLeave
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\VoiceExtractor.exe"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_FINISHPAGE_RUN_TEXT "启动 Voice Extractor"
!insertmacro MUI_PAGE_FINISH
!define MUI_UNCONFIRMPAGE_TEXT_TOP "将移除程序文件和快捷方式，保留模型、设置、日志和提取结果。"
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Section "程序和内置声纹模型（必选）" Main
    SectionIn 1 RO
    SetOverwrite on
    ClearErrors
    !include "${STAGE_DIR}\install-files.nsh"
    IfErrors 0 files_ready
        MessageBox MB_OK|MB_ICONSTOP "部分文件未能安装。请关闭正在运行的程序，并检查磁盘空间及安装目录权限。" /SD IDOK
        SetErrorLevel 1
        Abort
    files_ready:
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    WriteRegStr HKCU "Software\Voice-Extractor" "InstallDir" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "DisplayName" "Voice-Extractor1.0"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "DisplayVersion" "1.0"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "Publisher" "Voice Extractor contributors"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "DisplayIcon" "$INSTDIR\VoiceExtractor.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "EstimatedSize" ${INSTALLED_SIZE_KB}
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor" "NoRepair" 1
SectionEnd

Section "Python 与运行依赖" Runtime
    SectionIn 1
    AddSize 8388608
    StrCpy $RuntimeMode "install"
SectionEnd

Section "补齐全部提取模型" Models
    SectionIn 1
    AddSize 6291456
    StrCpy $ModelMode "install"
SectionEnd

Section "-完成环境安装与验证"
    InitPluginsDir
    SetOutPath "$PLUGINSDIR\bootstrap-python"
    File /r "${STAGE_DIR}\bootstrap\*"
    SetOutPath "$INSTDIR"
    configure_runtime:
        DetailPrint "正在安装并验证运行环境和全部模型，请勿关闭安装器。"
        nsExec::ExecToLog /TIMEOUT=1200000 '"$PLUGINSDIR\bootstrap-python\python.exe" -u "$INSTDIR\resource\setup\nsis_setup.py" --root "$INSTDIR" --runtime $RuntimeMode --models $ModelMode --device $Device'
        Pop $0
        ${If} $0 != 0
            MessageBox MB_RETRYCANCEL|MB_ICONSTOP "安装未完成。请查看上方详细信息或 $INSTDIR\logs\installer.log。$\r$\n$\r$\n修复网络或依赖问题后可重试，已完成的下载会复用。更改安装选项请取消并重新运行安装器。" /SD IDCANCEL IDRETRY configure_runtime
            SetErrorLevel 1
            Abort
        ${EndIf}
SectionEnd

Section "桌面快捷方式" DesktopShortcut
    SectionIn 1
    CreateShortcut "$DESKTOP\Voice Extractor.lnk" "$INSTDIR\VoiceExtractor.exe"
SectionEnd

Section "开始菜单快捷方式" StartMenuShortcut
    SectionIn 1
    CreateDirectory "$SMPROGRAMS\Voice-Extractor"
    CreateShortcut "$SMPROGRAMS\Voice-Extractor\Voice Extractor.lnk" "$INSTDIR\VoiceExtractor.exe"
    CreateShortcut "$SMPROGRAMS\Voice-Extractor\卸载.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${Main} "安装程序、FFmpeg、后台资源和内置声纹模型，保留已有强化模型。"
    !insertmacro MUI_DESCRIPTION_TEXT ${Runtime} "安装独立 Python、PyTorch 和提取依赖。取消勾选时复用安装目录中的环境，并检查其是否可用。"
    !insertmacro MUI_DESCRIPTION_TEXT ${Models} "补齐人声分离、说话人校验、重叠检测及语音识别模型。取消勾选时必须已具备全部模型。"
    !insertmacro MUI_DESCRIPTION_TEXT ${DesktopShortcut} "在当前用户桌面创建启动快捷方式。"
    !insertmacro MUI_DESCRIPTION_TEXT ${StartMenuShortcut} "在开始菜单创建启动和卸载快捷方式。"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInit
    ${IfNot} ${RunningX64}
        MessageBox MB_OK|MB_ICONSTOP "Voice Extractor 需要 64 位 Windows。" /SD IDOK
        Abort
    ${EndIf}
    SetRegView 64
    SetShellVarContext current
    StrCpy $RuntimeMode "reuse"
    StrCpy $ModelMode "reuse"
    StrCpy $Device "cpu"
    ${GetParameters} $Arguments
    ${GetOptions} $Arguments "/DEVICE=" $Option
    ${If} $Option == "cu118"
        StrCpy $Device "cu118"
    ${EndIf}
    ${GetOptions} $Arguments "/RUNTIME=" $Option
    ${If} $Option == "reuse"
        !insertmacro UnselectSection ${Runtime}
    ${EndIf}
    ${GetOptions} $Arguments "/MODELS=" $Option
    ${If} $Option == "reuse"
        !insertmacro UnselectSection ${Models}
    ${EndIf}
FunctionEnd

Function DevicePage
    !insertmacro MUI_HEADER_TEXT "选择计算环境" "安装器将根据选择安装对应依赖。"
    nsDialogs::Create 1018
    Pop $DeviceDialog
    ${NSD_CreateLabel} 0 0 100% 30u "CPU 兼容所有支持的电脑。NVIDIA GPU 模式需要兼容 CUDA 11.8 的显卡和驱动，下载量更大。"
    Pop $0
    ${NSD_CreateRadioButton} 0 40u 100% 14u "CPU 模式（通用）"
    Pop $CpuRadio
    ${NSD_CreateRadioButton} 0 65u 100% 14u "NVIDIA GPU 加速（CUDA 11.8）"
    Pop $CudaRadio
    ${If} $Device == "cu118"
        ${NSD_Check} $CudaRadio
    ${Else}
        ${NSD_Check} $CpuRadio
    ${EndIf}
    ${NSD_CreateLabel} 0 98u 100% 35u "安装器会验证运行环境和全部提取模型；只有检查通过才会进入完成页。无需单独安装 CUDA Toolkit。"
    Pop $0
    nsDialogs::Show
FunctionEnd

Function DevicePageLeave
    ${NSD_GetState} $CudaRadio $0
    ${If} $0 == ${BST_CHECKED}
        StrCpy $Device "cu118"
    ${Else}
        StrCpy $Device "cpu"
    ${EndIf}
FunctionEnd

Function DirectoryPageLeave
    ${IfNot} ${SectionIsSelected} ${Runtime}
        IfFileExists "$INSTDIR\resource\python\python.exe" runtime_present
        MessageBox MB_OK|MB_ICONEXCLAMATION "当前目录没有可复用的 Python，请返回勾选“Python 与运行依赖”。"
        Abort
        runtime_present:
    ${EndIf}
    ${IfNot} ${SectionIsSelected} ${Models}
        IfFileExists "$INSTDIR\voice_extractor_installation.json" models_present
        MessageBox MB_OK|MB_ICONEXCLAMATION "当前目录没有完整安装记录，请返回勾选“补齐全部提取模型”。"
        Abort
        models_present:
    ${EndIf}
FunctionEnd

Function un.onInit
    SetRegView 64
    SetShellVarContext current
FunctionEnd

Section "Uninstall"
    !include "${STAGE_DIR}\uninstall-files.nsh"
    RMDir /r "$INSTDIR\resource\python"
    RMDir /r "$INSTDIR\resource\python_packages"
    Delete "$INSTDIR\resource\runtime-constraints.txt"
    Delete "$INSTDIR\voice_extractor_installation.json"
    Delete "$DESKTOP\Voice Extractor.lnk"
    Delete "$SMPROGRAMS\Voice-Extractor\Voice Extractor.lnk"
    Delete "$SMPROGRAMS\Voice-Extractor\卸载.lnk"
    RMDir "$SMPROGRAMS\Voice-Extractor"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Voice-Extractor"
    DeleteRegKey HKCU "Software\Voice-Extractor"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"
SectionEnd
