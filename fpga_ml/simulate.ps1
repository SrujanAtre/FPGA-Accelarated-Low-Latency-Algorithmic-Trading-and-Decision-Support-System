$ErrorActionPreference = 'Stop'
$mlSimulationRoot = Join-Path $PSScriptRoot 'rtl'
Push-Location $mlSimulationRoot
try {
    & 'C:\Xilinx\Vivado\2024.2\bin\xvlog.bat' --include . ml_core.v tb_ml_core.v fpga_ml_top.v uart_rx.v uart_tx.v tb_uart.v
    if ($LASTEXITCODE) { throw 'Verilog compilation failed' }
    & 'C:\Xilinx\Vivado\2024.2\bin\xelab.bat' tb_ml_core -s ml_test
    if ($LASTEXITCODE) { throw 'Core elaboration failed' }
    & 'C:\Xilinx\Vivado\2024.2\bin\xsim.bat' ml_test -runall
    if ($LASTEXITCODE) { throw 'Core simulation failed' }
    & 'C:\Xilinx\Vivado\2024.2\bin\xelab.bat' tb_uart -L unisims_ver -s uart_test
    if ($LASTEXITCODE) { throw 'UART elaboration failed' }
    & 'C:\Xilinx\Vivado\2024.2\bin\xsim.bat' uart_test -runall
    if ($LASTEXITCODE) { throw 'UART simulation failed' }
} finally { Pop-Location }
