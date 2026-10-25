set root [file normalize [file dirname [info script]]]
open_checkpoint "$root/reports/ml_routed.dcp"
read_saif -strip_path tb_uart/dut "$root/rtl/uart_activity.saif"
report_power -file "$root/reports/power_uart_activity.rpt"
report_utilization -hierarchical -file "$root/reports/utilization_hierarchical.rpt"
report_clock_utilization -file "$root/reports/clocks.rpt"
