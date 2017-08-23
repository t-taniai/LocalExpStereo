echo off

set bin=%~dp0x64\Release\LocalExpansionStereo.exe
set datasetroot=%~dp0data
set resultsroot=%~dp0results

mkdir "%resultsroot%"
"%bin%" -targetDir "%datasetroot%\MiddV2\cones" -outputDir "%resultsroot%\cones" -mode MiddV2 -smooth_weight 1 -doDual 1
"%bin%" -targetDir "%datasetroot%\MiddV2\teddy" -outputDir "%resultsroot%\teddy" -mode MiddV2 -smooth_weight 1
"%bin%" -targetDir "%datasetroot%\MiddV3\Adirondack" -outputDir "%resultsroot%\Adirondack" -mode MiddV3 -smooth_weight 0.5
