# ============================================================
# ZCU104 125 MHz PL CLOCK
# ============================================================

set_property PACKAGE_PIN F23 [get_ports CLK_125_P]
set_property IOSTANDARD LVDS [get_ports CLK_125_P]

set_property PACKAGE_PIN E23 [get_ports CLK_125_N]
set_property IOSTANDARD LVDS [get_ports CLK_125_N]

create_clock -period 8.000 \
    -name clk_125 \
    -waveform {0.000 4.000} \
    [get_ports CLK_125_P]


# ============================================================
# ZCU104 PL UART
# ============================================================

# A20 = FPGA RX
set_property PACKAGE_PIN A20 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx]

# C19 = FPGA TX
set_property PACKAGE_PIN C19 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx]